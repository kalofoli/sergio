"""Generic socket server classes.
This module tries to capture the various aspects of defining a server:
For socket-based servers:
- address family:
        - AF_INET{,6}: IP (Internet Protocol) sockets (default)
        - AF_UNIX: Unix domain sockets
        - others, e.g. AF_DECNET are conceivable (see <socket.h>
- socket type:
        - SOCK_STREAM (reliable stream, e.g. TCP)
        - SOCK_DGRAM (datagrams, e.g. UDP)
For request-based servers (including socket-based):
- client address verification before further looking at the request
        (This is actually a hook for any processing that needs to look
         at the request before anything else, e.g. logging)
- how to handle multiple requests:
        - synchronous (one request is handled at a time)
        - forking (each request is handled by a new process)
        - threading (each request is handled by a new thread)
The classes in this module favor the server type that is simplest to
write: a synchronous TCP/IP server.  This is bad class design, but
save some typing.  (There's also the issue that a deep class hierarchy
slows down method lookups.)
There are five classes in an inheritance diagram, four of which represent
synchronous servers of four types:
        +------------+
        | CoreServer |
        +------------+
              |
              v
        +------------+
        | BaseServer |
        +------------+
              |
              v
        +-----------+        +------------------+
        | TCPServer |------->| UnixStreamServer |
        +-----------+        +------------------+
              |
              v
        +-----------+        +--------------------+
        | UDPServer |------->| UnixDatagramServer |
        +-----------+        +--------------------+
Note that UnixDatagramServer derives from UDPServer, not from
UnixStreamServer -- the only difference between an IP and a Unix
stream server is the address family, which is simply repeated in both
unix server classes.
Forking and threading versions of each type of server can be created
using the ForkingMixIn and ThreadingMixIn mix-in classes.  For
instance, a threading UDP server class is created as follows:
        class ThreadingUDPServer(ThreadingMixIn, UDPServer): pass
The Mix-in class must come first, since it overrides a method defined
in UDPServer! Setting the various member variables also changes
the behavior of the underlying server mechanism.
To implement a service, you must derive a class from
BaseRequestHandler and redefine its handle() method.  You can then run
various versions of the service by combining one of the server classes
with your request handler class.
The request handler class must be different for datagram or stream
services.  This can be hidden by using the request handler
subclasses StreamRequestHandler or DatagramRequestHandler.
Of course, you still have to use your head!
For instance, it makes no sense to use a forking server if the service
contains state in memory that can be modified by requests (since the
modifications in the child process would never reach the initial state
kept in the parent process and passed to each child).  In this case,
you can use a threading server, but you will probably have to use
locks to avoid two requests that come in nearly simultaneous to apply
conflicting changes to the server state.
On the other hand, if you are building e.g. an HTTP server, where all
data is stored externally (e.g. in the file system), a synchronous
class will essentially render the service "deaf" while one request is
being handled -- which may be for a very long time if a client is slow
to read all the data it has requested.  Here a threading or forking
server is appropriate.
In some cases, it may be appropriate to process part of a request
synchronously, but to finish processing in a forked child depending on
the request data.  This can be implemented by using a synchronous
server and doing an explicit fork in the request handler class
handle() method.
Another approach to handling multiple simultaneous requests in an
environment that supports neither threads nor fork (or where these are
too expensive or inappropriate for the service) is to maintain an
explicit table of partially finished requests and to use a selector to
decide which request to work on next (or whether to handle a new
incoming request).  This is particularly important for stream services
where each client can potentially be connected for a long time (if
threads or subprocesses cannot be used).
Future work:
- Standard classes for Sun RPC (which uses either UDP or TCP)
- Standard mix-in classes to implement various authentication
  and encryption schemes
XXX Open problems:
- What to do with out-of-band data?
BaseServer:
- split generic "request" functionality out into BaseServer class.
  Copyright (C) 2000  Luke Kenneth Casson Leighton <lkcl@samba.org>
  example: read entries from a SQL database (requires overriding
  get_request() to return a table entry from the database).
  entry is processed by a RequestHandlerClass.

Created on Sep 10, 2018

@author: janis (for the edits) based on cpython's socketserver
"""

# Author of the BaseServer patch: Luke Kenneth Casson Leighton

# pylint: disable=multiple-statements

__version__ = "0.4"


import socket
import selectors
import io
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
import enum
from errno import ECONNRESET
import select

__all__ = ["BaseServer", "TCPServer", "UDPServer",
           "ThreadingUDPServer", "ThreadingTCPServer",
           "BaseRequestHandler", "StreamRequestHandler",
           "DatagramRequestHandler", "ThreadingMixIn"]
if hasattr(os, "fork"):
    __all__.extend(["ForkingUDPServer","ForkingTCPServer", "ForkingMixIn"])
if hasattr(socket, "AF_UNIX"):
    __all__.extend(["UnixStreamServer","UnixDatagramServer",
                    "ThreadingUnixStreamServer",
                    "ThreadingUnixDatagramServer"])

# poll/select have the advantage of not requiring any extra file descriptor,
# contrarily to epoll/kqueue (also, they require a single syscall).
if hasattr(selectors, 'PollSelector'):
    _ServerSelector = selectors.PollSelector
else:
    _ServerSelector = selectors.SelectSelector

class StartStopMixin:
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._main_thread = None
        
    def start(self, name=None, *args):
        '''Create a thread that will invoke the serve_forever loop.
        
        @param name: the name o fthe main thread.
        '''
        self._main_thread = threading.Thread(name=name,
                                             target=self.serve_forever,
                                             args=args)
        self._main_thread.start()

    def stop(self):
        '''Request the main loop to stop and wait for its termination.'''
        if self._main_thread is None:
            raise ValueError(f'No main thread running for this server.')
        self.initiate_shutdown()
        self._main_thread.join()
        self._main_thread = None
    
    def wait(self, timeout=None):
        '''Wait for the main server thread to finish.'''
        if self._main_thread is None:
            raise ValueError(f'Thread not started')
        return self._main_thread.join(timeout=timeout)
    
class ServerState(enum.Enum):
    IDLE = enum.auto()
    BOUND = enum.auto()
    ACTIVATED = enum.auto()
    RUNNING = enum.auto()
    STOPPING = enum.auto()
    CLOSED = enum.auto()

class RequestInfo:
    def __init__(self, server, request, client_address, socket=None, handler = None):
        self.server = server
        self.request = request
        self.handler = handler
        self.client_address = client_address
        self.socket = socket if socket is not None else server.get_socket_from_request(request)

class CoreServer:

    """Base class for server classes.
    Methods for the caller:
    - __init__(server_address, RequestHandlerClass)
    - serve_forever(poll_interval=0.5)
    - shutdown()
    - handle_request()  # if you do not use serve_forever()
    - fileno() -> int   # for selector
    Methods that may be overridden:
    - server_bind()
    - server_activate()
    - get_request() -> request, client_address
    - handle_timeout()
    - verify_request(request, client_address)
    - server_close()
    - process_request(request, client_address)
    - shutdown_request(request)
    - close_request(request)
    - service_actions()
    - handle_error()
    Methods for derived classes:
    - finish_request(request, client_address)
    Class variables that may be overridden by derived classes or
    instances:
    - timeout
    - address_family
    - socket_type
    - allow_reuse_address
    Instance variables:
    - RequestHandlerClass
    - socket
    """

    timeout = None

    def __init__(self, server_address, RequestHandlerClass):
        """Constructor.  May be extended, do not override."""
        self.server_address = server_address
        self.RequestHandlerClass = RequestHandlerClass
        self.shutdown_requested = False
        self._main_thread = None
        self.infos = {}
        self.state = ServerState.IDLE
        self.__is_shut_down = threading.Event()

    def get_request_info(self, request):
        """Get the handler for the given request"""
        sock = self.get_socket_from_request(request)
        return self.infos.get(sock.fileno(), None)

    def server_bind(self):
        """Called by constructor to bind the server.
        May be overridden.
        """
        self.state = ServerState.BOUND
    
    def server_activate(self):
        """Called by constructor to activate the server.
        May be overridden.
        """
        self.state = ServerState.ACTIVATED

    def initiate_shutdown(self):
        """Requests the serve_forever loop to terminate, but does not wait.
        """
        self.shutdown_requested = True
        self.state = ServerState.STOPPING
        
    def service_actions(self):
        """Called by the serve_forever() loop.
        May be overridden by a subclass / Mixin to implement any code that
        needs to be run during the loop.
        """
        pass

    # The distinction between handling, getting, processing and finishing a
    # request is fairly arbitrary.  Remember:
    #
    # - handle_request() is the top-level call.  It calls selector.select(),
    #   get_request(), verify_request() and process_request()
    # - get_request() is different for stream or datagram sockets
    # - process_request() is the place that may fork a new process or create a
    #   new thread to finish the request
    # - finish_request() instantiates the request handler class; this
    #   constructor will handle the request all by itself

    def handle_request(self):
        """Handle one request, possibly blocking.
        Respects self.timeout.
        """
        pass
    
    def serve_forever_started(self):
        """Called when the server starts running.
        """
        self.__is_shut_down.clear()
        self.state = ServerState.RUNNING
    
    def serve_forever_stopped(self):
        """Called when the server starts running.
        """
        self.shutdown_requested = False
        self.state = ServerState.IDLE
        self.__is_shut_down.set()
        
    def handle_timeout(self):
        """Called if no new request arrives within self.timeout.
        Overridden by ForkingMixIn.
        """
        pass

    def verify_request(self, request, client_address):
        """Verify the request.  May be overridden.
        Return True if we should proceed with this request.
        """
        return True

    def process_request(self, request, client_address):
        """Call finish_request.
        Overridden by ForkingMixIn and ThreadingMixIn.
        """
        self.finish_request(request, client_address)
        self.shutdown_request(request)

    def server_close(self):
        """Called to clean-up the server.
        May be overridden.
        """
        self.state = ServerState.CLOSED

    def shutdown(self):
        """Stops the serve_forever loop.
        Blocks until the loop has finished. This must be called while
        serve_forever() is running in another thread, or it will
        deadlock.
        """
        self.initiate_shutdown()
        self.__is_shut_down.wait()
        self.server_close()

    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass."""
        handler = self.RequestHandlerClass(request, client_address, self)
        infos = self.get_request_info(request)
        infos.handler = handler
        handler.run()
        self.request_completed(request)
    
    def request_completed(self, request):
        """Called when a request has finished"""

    def shutdown_request(self, request):
        """Called to shutdown and close an individual request."""
        self.close_request(request)

    def close_request(self, request):
        """Called to clean up an individual request."""
        sock = self.get_socket_from_request(request)
        del self.infos[sock.fileno()]

    def handle_error(self, request, client_address):
        """Handle an error gracefully.  May be overridden.
        The default is to print a traceback and continue.
        """
        print('-'*40, file=sys.stderr)
        print('Exception happened during processing of request from',
            client_address, file=sys.stderr)
        import traceback
        traceback.print_exc()
        print('-'*40, file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server_close()


class BaseServer(CoreServer):

    """Base class for server classes.
    Methods for the caller:
    - __init__(server_address, RequestHandlerClass)
    - serve_forever(poll_interval=0.5)
    - shutdown()
    - handle_request()  # if you do not use serve_forever()
    - fileno() -> int   # for selector
    Methods that may be overridden:
    - server_bind()
    - server_activate()
    - get_request() -> request, client_address
    - handle_timeout()
    - verify_request(request, client_address)
    - server_close()
    - process_request(request, client_address)
    - shutdown_request(request)
    - close_request(request)
    - service_actions()
    - handle_error()
    Methods for derived classes:
    - finish_request(request, client_address)
    Class variables that may be overridden by derived classes or
    instances:
    - timeout
    - address_family
    - socket_type
    - allow_reuse_address
    Instance variables:
    - RequestHandlerClass
    - socket
    """

    def __init__(self, server_address, RequestHandlerClass):
        """Constructor.  May be extended, do not override."""
        super().__init__(server_address, RequestHandlerClass)

    def serve_forever(self, poll_interval=0.5):
        """Handle one request at a time until shutdown.
        Polls for shutdown every poll_interval seconds. Ignores
        self.timeout. If you need to do periodic tasks, do them in
        another thread.
        """
        self.serve_forever_started()
        try:
            # XXX: Consider using another file descriptor or connecting to the
            # socket to wake this up instead of polling. Polling reduces our
            # responsiveness to a shutdown request and wastes cpu at all other
            # times.
            with _ServerSelector() as selector:
                selector.register(self, selectors.EVENT_READ)

                while not self.shutdown_requested:
                    ready = selector.select(poll_interval)
                    if ready:
                        self._handle_request_noblock()

                    self.service_actions()
        finally:
            self.serve_forever_stopped()

    def handle_request(self):
        """Handle one request, possibly blocking.
        Respects self.timeout.
        """
        # Support people who used socket.settimeout() to escape
        # handle_request before self.timeout was available.
        timeout = self.socket.gettimeout()
        if timeout is None:
            timeout = self.timeout
        elif self.timeout is not None:
            timeout = min(timeout, self.timeout)
        if timeout is not None:
            deadline = time() + timeout

        # Wait until a request arrives or the timeout expires - the loop is
        # necessary to accommodate early wakeups due to EINTR.
        with _ServerSelector() as selector:
            selector.register(self, selectors.EVENT_READ)

            while True:
                ready = selector.select(timeout)
                if ready:
                    return self._handle_request_noblock()
                else:
                    if timeout is not None:
                        timeout = deadline - time()
                        if timeout < 0:
                            return self.handle_timeout()

    def _handle_request_noblock(self):
        """Handle one request, without blocking.
        I assume that selector.select() has returned that the socket is
        readable before this function was called, so there should be no risk of
        blocking in get_request().
        """
        try:
            request, client_address = self.get_request()
            info = RequestInfo(server=self, request=request, client_address=client_address)
            self.info[info.socket.fileno()] = info
        except OSError:
            return
        if self.verify_request(request, client_address):
            try:
                self.process_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
                self.shutdown_request(request)
            except:
                self.shutdown_request(request)
                raise
        else:
            self.shutdown_request(request)

    def handle_timeout(self):
        """Called if no new request arrives within self.timeout.
        Overridden by ForkingMixIn.
        """
        pass

    def process_request(self, request, client_address):
        """Call finish_request.
        Overridden by ForkingMixIn and ThreadingMixIn.
        """
        self.finish_request(request, client_address)
        self.shutdown_request(request)

    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass."""
        handler = self.RequestHandlerClass(request, client_address, self)
        infos = self.get_request_info(request)
        infos.handler = handler
        handler.run()


class TCPMixin:

    """Base class for various socket-based server classes.
    Defaults to synchronous IP stream (i.e., TCP).
    Methods for the caller:
    - __init__(server_address, RequestHandlerClass, bind_and_activate=True)
    - serve_forever(poll_interval=0.5)
    - shutdown()
    - handle_request()  # if you don't use serve_forever()
    - fileno() -> int   # for selector
    Methods that may be overridden:
    - server_bind()
    - server_activate()
    - get_request() -> request, client_address
    - handle_timeout()
    - verify_request(request, client_address)
    - process_request(request, client_address)
    - shutdown_request(request)
    - close_request(request)
    - handle_error()
    Methods for derived classes:
    - finish_request(request, client_address)
    Class variables that may be overridden by derived classes or
    instances:
    - timeout
    - address_family
    - socket_type
    - request_queue_size (only for stream sockets)
    - allow_reuse_address
    Instance variables:
    - server_address
    - RequestHandlerClass
    - socket
    """

    address_family = socket.AF_INET

    socket_type = socket.SOCK_STREAM

    request_queue_size = 5

    allow_reuse_address = False

    def __init__(self, *args, bind_and_activate=True, **kwargs):
        """Constructor.  May be extended, do not override."""
        super().__init__(*args, **kwargs)
        self.socket = socket.socket(self.address_family,
                                    self.socket_type)
        if bind_and_activate:
            try:
                self.server_bind()
                self.server_activate()
            except:
                self.server_close()
                raise

    def server_bind(self):
        """Called by constructor to bind the socket.
        May be overridden.
        """
        if self.allow_reuse_address:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        super().server_bind()

    def server_activate(self):
        """Called by constructor to activate the server.
        May be overridden.
        """
        self.socket.listen(self.request_queue_size)
        super().server_activate()

    def server_close(self):
        """Called to clean-up the server.
        May be overridden.
        """
        self.socket.close()
        super().server_close()

    def fileno(self):
        """Return socket file number.
        Interface required by selector.
        """
        return self.socket.fileno()

    def get_request(self):
        """Get the request and client address from the socket.
        May be overridden.
        """
        return self.socket.accept()

    @classmethod
    def get_socket_from_request(cls, request):
        return request

    def shutdown_request(self, request):
        """Called to shutdown and close an individual request."""
        try:
            #explicitly shutdown.  socket.close() merely releases
            #the socket and waits for GC to perform the actual close.
            request.shutdown(socket.SHUT_WR)
        except OSError:
            pass #some platforms may raise ENOTCONN here
        super().shutdown_request(request)

    def close_request(self, request):
        """Called to clean up an individual request."""
        super().close_request(request)
        request.close()

class TCPServer(TCPMixin, BaseServer): pass

class ServerShuttingDownError(Exception):
    pass

class WaitableEvent:
    """Provides an abstract object that can be used to resume select loops with
    indefinite waits from another thread or process. This mimics the standard
    threading.Event interface."""
    def __init__(self):
        self._read_fd, self._write_fd = os.pipe()

    def wait(self, timeout=None):
        rfds, wfds, efds = select.select([self._read_fd], [], [], timeout)
        return self._read_fd in rfds

    def isSet(self):
        return self.wait(0)

    def clear(self):
        if self.isSet():
            os.read(self._read_fd, 1)

    def set(self):
        if not self.isSet():
            os.write(self._write_fd, b'1')

    def fileno(self):
        """Return the FD number of the read side of the pipe, allows this object to
        be used with select.select()."""
        return self._read_fd

    def __del__(self):
        os.close(self._read_fd)
        os.close(self._write_fd)
        
class AsyncBaseServer(CoreServer):
    backlog = 1
    timeout = None
    SelectorClass = selectors.DefaultSelector
    def __init__(self, address, RequestHandlerClass):
        super().__init__(address, RequestHandlerClass)
        self.selector = self.SelectorClass()
        self.__service_event = WaitableEvent()
        self.service_events = []
        self.selector.register(self.__service_event, selectors.EVENT_READ, 'service_event')

    def trigger_service_event(self, event):
        self.service_events.append(event)
        self.__service_event.set()
    
    def initiate_shutdown(self):
        try:
            self.selector.unregister(self.socket)
        except KeyError: pass
        try:
            self.socket.shutdown(socket.SHUT_RD)
        except OSError: pass
        self.trigger_service_event('terminate_handlers')
        super().initiate_shutdown()
    
    def shutdown(self):
        super().shutdown()

    def server_activate(self):
        super().server_activate()
        self.selector.register(self.socket, selectors.EVENT_READ, 'accept')

    def serve_forever(self):
        self.serve_forever_started()
        while self.state in {ServerState.RUNNING, ServerState.STOPPING}:
            if self.state == ServerState.STOPPING and not self.infos:
                break
            events = self.selector.select(self.timeout)
            for key,mask in events:  # pylint: disable=unused-variable
                sock,fd,mask,data = key # pylint: disable=unused-variable
                if data == 'accept':
                    try:
                        if ServerState.RUNNING:
                            request, client_address = self.get_request()
                            client_socket = self.get_socket_from_request(request)
                            client_socket.setblocking(False)
                            info = RequestInfo(server=self, request=request, socket=client_socket, client_address=client_address)
                            self.infos[client_socket.fileno()] = info
                            if self.verify_request(request, client_address):
                                handler = self.RequestHandlerClass(request, client_address, self)
                                info.handler = handler
                                self.selector.register(client_socket, selectors.EVENT_READ, info)
                                handler.setup()
                            else:
                                self.shutdown_request(request)
                        else:
                            raise ServerShuttingDownError(f'Server is shutting down.')
                    except OSError:
                        self.handle_error(request)
                elif isinstance(data, RequestInfo):
                    self.data_available(data)
                elif data == 'service_event':
                    self.handle_service_events()
                        
        self.serve_forever_stopped()
    
    def data_available(self, data):
        request = data.request
        handler = data.handler
        client_address = data.client_address
        try:
            should_stop = handler.data_available()
            if should_stop or handler.stop_requested:
                self.request_completed(request)
                self.shutdown_request(request)
        except Exception:
            self.handle_error(request, client_address)
            self.shutdown_request(request)
        except:
            self.shutdown_request(request)
            raise

    def handle_service_events(self):
        """Handle pevents sent to the server from other threads.
        
        Used for the terminate event, so that the handlers are asked to 
        terminate and are given a chance to execute a cleanup
        (or even ignore the termination request) in the server thread"""
        self.__service_event.clear()
        while self.service_events:
            event = self.service_events[0]
            self.handle_service_event(event)
            self.service_events = self.service_events[1:]

    def handle_service_event(self, event):
        """Handle a service event. Called to terminate handlers.
        
        This is run by the serving thread."""
        if event == 'terminate_handlers':
            for data in tuple(self.infos.values()):
                data.handler.request_stop()
                self.data_available(data)
    
    def shutdown_request(self, request):
        try:
            info = self.get_request_info(request)
            if info.handler is not None:
                info.handler.finish()
        except:
            self.handle_error(request, 'During finish')
        super().shutdown_request(request)

    def close_request(self, request):
        sock = self.get_socket_from_request(request)
        try:
            self.selector.unregister(sock)
        except KeyError:pass
        super().close_request(request)
    


    def __repr__(self):
        return f'<{self.__class__.__name__} at {self.server_address!r} [{self.state.name}] with {len(self.infos)} connections>'



class SelectingTCPServer(TCPMixin, AsyncBaseServer):
    pass

class UDPServer(TCPServer):

    """UDP server class."""

    allow_reuse_address = False

    socket_type = socket.SOCK_DGRAM

    max_packet_size = 8192

    def get_request(self):
        data, client_addr = self.socket.recvfrom(self.max_packet_size)
        return (data, self.socket), client_addr
    
    @classmethod
    def get_socket_from_request(cls, request):
        return request[1]

    def server_activate(self):
        # No need to call listen() for UDP.
        pass

    def shutdown_request(self, request):
        # No need to shutdown anything.
        self.close_request(request)

    def close_request(self, request):
        # No need to close anything.
        pass

if hasattr(os, "fork"):
    class ForkingMixIn:
        """Mix-in class to handle each request in a new process."""

        timeout = 300
        active_children = None
        max_children = 40
        # If true, server_close() waits until all child processes complete.
        block_on_close = True

        def collect_children(self, *, blocking=False):
            """Internal routine to wait for children that have exited."""
            if self.active_children is None:
                return

            # If we're above the max number of children, wait and reap them until
            # we go back below threshold. Note that we use waitpid(-1) below to be
            # able to collect children in size(<defunct children>) syscalls instead
            # of size(<children>): the downside is that this might reap children
            # which we didn't spawn, which is why we only resort to this when we're
            # above max_children.
            while len(self.active_children) >= self.max_children:
                try:
                    pid, _ = os.waitpid(-1, 0)
                    self.active_children.discard(pid)
                except ChildProcessError:
                    # we don't have any children, we're done
                    self.active_children.clear()
                except OSError:
                    break

            # Now reap all defunct children.
            for pid in self.active_children.copy():
                try:
                    flags = 0 if blocking else os.WNOHANG
                    pid, _ = os.waitpid(pid, flags)
                    # if the child hasn't exited yet, pid will be 0 and ignored by
                    # discard() below
                    self.active_children.discard(pid)
                except ChildProcessError:
                    # someone else reaped it
                    self.active_children.discard(pid)
                except OSError:
                    pass

        def handle_timeout(self):
            """Wait for zombies after self.timeout seconds of inactivity.
            May be extended, do not override.
            """
            self.collect_children()

        def service_actions(self):
            """Collect the zombie child processes regularly in the ForkingMixIn.
            service_actions is called in the BaseServer's serve_forver loop.
            """
            self.collect_children()

        def process_request(self, request, client_address):
            """Fork a new subprocess to process the request."""
            pid = os.fork()
            if pid:
                # Parent process
                if self.active_children is None:
                    self.active_children = set()
                self.active_children.add(pid)
                self.close_request(request)
                return
            else:
                # Child process.
                # This must never return, hence os._exit()!
                status = 1
                try:
                    self.finish_request(request, client_address)
                    status = 0
                except Exception:
                    self.handle_error(request, client_address)
                finally:
                    try:
                        self.shutdown_request(request)
                    finally:
                        os._exit(status)

        def server_close(self):
            super().server_close()
            self.collect_children(blocking=self.block_on_close)


class ThreadingMixIn:
    """Mix-in class to handle each request in a new thread."""

    # Decides how threads will act upon termination of the
    # main process
    daemon_threads = False
    # If true, server_close() waits until all non-daemonic threads terminate.
    block_on_close = True
    # For non-daemonic threads, list of threading.Threading objects
    # used by server_close() to wait for all threads completion.
    _threads = None

    def process_request_thread(self, request, client_address):
        """Same as in BaseServer but as a thread.
        In addition, exception handling is done here.
        """
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def process_request(self, request, client_address):
        """Start a new thread to process the request."""
        t = threading.Thread(target = self.process_request_thread,
                             args = (request, client_address))
        t.daemon = self.daemon_threads
        if not t.daemon and self.block_on_close:
            if self._threads is None:
                self._threads = []
            self._threads.append(t)
        t.start()

    def server_close(self):
        super().server_close()
        if self.block_on_close:
            threads = self._threads
            self._threads = None
            if threads:
                for thread in threads:
                    thread.join()


if hasattr(os, "fork"):
    class ForkingUDPServer(ForkingMixIn, UDPServer): pass
    class ForkingTCPServer(ForkingMixIn, TCPServer): pass

class ThreadingUDPServer(ThreadingMixIn, UDPServer): pass
class ThreadingTCPServer(ThreadingMixIn, TCPServer): pass

if hasattr(socket, 'AF_UNIX'):

    class UnixMixin:
        address_family = socket.AF_UNIX
        
        unlink_socket_on_close = True
        unlink_socket_on_start = False
        def server_bind(self):
            if self.unlink_socket_on_start and os.path.exists(self.server_address):
                os.unlink(self.server_address)
            super().server_bind()
    
        def server_closed(self):
            super().server_closed()
            if self.unlink_socket_on_close and os.path.exists(self.server_address):
                os.unlink(self.server_address)
    
    class SelectingUnixStreamServer(UnixMixin, SelectingTCPServer): pass
    
    class UnixStreamServer(UnixMixin, TCPServer): pass

    class UnixDatagramServer(UnixMixin, UDPServer): pass

    class ThreadingUnixStreamServer(ThreadingMixIn, UnixStreamServer): pass

    class ThreadingUnixDatagramServer(ThreadingMixIn, UnixDatagramServer): pass

class BaseRequestHandler:

    """Base class for request handler classes.
    This class is instantiated for each request to be handled.  The
    constructor sets the instance variables request, client_address
    and server, and then calls the handle() method.  To implement a
    specific service, all you need to do is to derive a class which
    defines a handle() method.
    The handle() method can find the request as self.request, the
    client address as self.client_address, and the server (in case it
    needs access to per-server information) as self.server.  Since a
    separate instance is created for each request, the handle() method
    can define other arbitrary instance variables.
    """

    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.fileno = request.fileno()

    def run(self):
        self.setup()
        try:
            self.handle()
        finally:
            self.finish()
        
    def setup(self):
        pass

    def handle(self):
        pass

    def finish(self):
        pass



# The following two classes make it possible to use the same service
# class for stream or datagram servers.
# Each class sets up these instance variables:
# - rfile: a file object from which receives the request is read
# - wfile: a file object to which the reply is written
# When the handle() method returns, wfile is flushed properly


class _SocketWriter(BufferedIOBase):
    """Simple writable BufferedIOBase implementation for a socket
    Does not hold data in a buffer, avoiding any need to call flush()."""

    def __init__(self, sock, mode='wb', encoding=None, errors='ignore'):
        self._sock = sock
        self.encoding = sys.getdefaultencoding() if encoding is None else encoding
        self.errors = errors
        self.binary = 'b' in mode
        assert mode in {"wb","w"}, "Only w nd wb modes supported."

    def writable(self):
        return True

    def write(self, data):
        b = data if self.binary else data.encode(self.encoding, errors=self.errors)
        self._sock.sendall(b)
        with memoryview(b) as view:
            return view.nbytes

    def fileno(self):
        return self._sock.fileno()

class StreamRequestMixin:

    """Define self.rfile and self.wfile for stream sockets."""

    # Default buffer sizes for rfile, wfile.
    # We default rfile to buffered because otherwise it could be
    # really slow for large data (a getc() call per byte); we make
    # wfile unbuffered because (a) often after a write() we want to
    # read and we need to flush the line; (b) big writes to unbuffered
    # files are typically optimized by stdio even when big reads
    # aren't.
    binary_streams = False
    rbufsize = -1
    wbufsize = 0
    encoding = None
    encoding_errors = 'ignore'
    newline = None
    # If true, a read on a closed file will raise a SocketClosed error
    # instead of silently returning no data. This is necessary for
    # asynchronous sockets, on which returning no data can mean that
    # otherwise the operation would block. 
    
    read_on_closed_raises = False

    # A timeout to apply to the request socket, if not None.
    timeout = None

    # Disable nagle algorithm for this socket, if True.
    # Use only when wbufsize != 0, to avoid small packets.
    disable_nagle_algorithm = False


    def setup(self):
        self.connection = self.request
        if self.timeout is not None:
            self.connection.settimeout(self.timeout)
        if self.disable_nagle_algorithm:
            self.connection.setsockopt(socket.IPPROTO_TCP,
                                       socket.TCP_NODELAY, True)
        if self.read_on_closed_raises:
            make_rfile = StreamRequestMixin._make_rfile
        else:
            make_rfile = socket.socket.makefile
        self.rfile = make_rfile(self.connection,
                                "rb" if self.binary_streams else "r",
                                buffering = self.rbufsize,
                                encoding=self.encoding,
                                errors=self.encoding_errors,
                                newline=self.newline
                                )
        if self.wbufsize == 0:
            make_wfile = _SocketWriter
            kwargs = {}
            assert self.newline is None, 'Cannot use newline translations with unbuffered writer'
        else:
            make_wfile = socket.socket.makefile
            kwargs = {'newline':self.newline,'buffering': self.wbufsize}

        self.wfile = make_wfile(self.connection,
                                mode="wb" if self.binary_streams else "w",
                                encoding=self.encoding,
                                errors=self.encoding_errors,
                                **kwargs
                                )

    def finish(self):
        if not self.wfile.closed:
            try:
                self.wfile.flush()
            except socket.error:
                # A final socket error may have occurred here, such as
                # the local error ECONNABORTED.
                pass
        self.wfile.close()
        self.rfile.close()

    @classmethod
    def _make_rfile(cls, socket, mode="r", buffering=None, *,
                     encoding=None, errors=None, newline=None):
        """makefile(...) -> an I/O stream connected to the socket
        The arguments are as for io.open() after the filename, except the only
        supported mode values are 'r' (default), 'w' and 'b'.
        """
        # XXX refactor to share code?
        assert mode in {"r","rb"}
        binary = "b" in mode
        raw = _AsyncSocketIO(socket, "r")
        if buffering is None:
            buffering = -1
        if buffering < 0:
            buffering = io.DEFAULT_BUFFER_SIZE
        if buffering == 0:
            if not binary:
                raise ValueError("unbuffered streams must be binary")
            return raw
        buffer = io.BufferedReader(raw, buffering)
        if binary:
            return buffer
        text = io.TextIOWrapper(buffer, encoding, errors, newline)
        text.mode = mode
        return text


class StreamRequestHandler(StreamRequestMixin, BaseRequestHandler):
    """Define self.rfile and self.wfile for stream sockets."""


class DatagramRequestHandler(BaseRequestHandler):

    """Define self.rfile and self.wfile for datagram sockets."""

    def setup(self):
        from io import BytesIO
        self.packet, self.socket = self.request
        self.rfile = BytesIO(self.packet)
        self.wfile = BytesIO()

    def finish(self):
        self.socket.sendto(self.wfile.getvalue(), self.client_address)

class StoppableRequestMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_requested = False

    def request_stop(self):
        """Ask the handler to gracefully stop.
        
        This is called by the server.
        To stop a request from the handler of the request handler,
        one must return True from the data_available() function."""
        self.stop_requested = True

class _BaseAsyncRequestHandler:

    """Base class for asynchronous request handler classes.
    This class is instantiated for each request to be handled.  The
    constructor sets the instance variables request, client_address
    and server, and then calls the handle() method.  To implement a
    specific service, all you need to do is to derive a class which
    defines a handle() method.
    The handle() method can find the request as self.request, the
    client address as self.client_address, and the server (in case it
    needs access to per-server information) as self.server.  Since a
    separate instance is created for each request, the handle() method
    can define other arbitrary instance variables.
    The handle method is invoked multiple times, whenever data is
    available to read. To designate a server termination, the
    method request_stop() must be called.
    """

    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.fileno = request.fileno()

    def setup(self):
        pass

    def data_available(self):
        pass

    def finish(self):
        """The server calls this upon completion"""
        pass
    
class BaseAsyncRequestHandler(StoppableRequestMixin, _BaseAsyncRequestHandler): pass

class SocketClosedError(OSError):
    def __init__(self, msg = 'Socket closed'):
        super().__init__(ECONNRESET, msg)

class _AsyncSocketIO(socket.SocketIO):
    def readinto(self, b):
        res = super().readinto(b)
        if res == 0:
            raise SocketClosedError()
        return res

class AsyncStreamRequestHandler(StreamRequestMixin, BaseAsyncRequestHandler):
    """Define self.rfile and self.wfile for async stream sockets."""
    read_on_closed_raises = True

class LineRequestMixin:
    read_on_close_raises = True
    handler_is_generator = None
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)
        if self.handler_is_generator is None:
            from inspect import isgeneratorfunction
            self.__class__.handler_is_generator = isgeneratorfunction(self.__class__.line_available)
    
    def setup(self):
        super().setup()
        self.generator = None
        if self.handler_is_generator:
            self.generator = self.line_available()
            next(self.generator)
    
    def __notify_generator(self, line):
        try:
            should_stop = self.generator.send(line)
        except StopIteration:
            should_stop = True
        return should_stop

    def notify_line(self, line):
        """Called by the RequestHandler when a new line is available.
        
        Used as an adaptor in case a generator is used as line_available.
        """ 
        if self.handler_is_generator:
            return self.__notify_generator(line)
        else:
            return self.line_available(line)

    def line_available(self, line):
        """This function must be overriden to handle lines"""
        pass
    
    def data_available(self):
        if self.stop_requested:
            if self.notify_line(False):
                return True
        try:
            for line in self.rfile:
                should_stop = self.notify_line(line)
                if should_stop or self.stop_requested:
                    return True
        except SocketClosedError:
            e_type, e_value, e_tb = sys.exc_info()
            self.socket_closed(e_type, e_value, e_tb)
    
    def socket_closed(self, exc_type, exc_value, exc_traceback):
        """Can be overriden.
        
        Called when socket is closed."""
        raise (exc_type, exc_value, exc_traceback)

class AsyncLineRequestHandler(LineRequestMixin, AsyncStreamRequestHandler): pass

class LineRequestHandler(LineRequestMixin, StoppableRequestMixin, StreamRequestHandler):
    read_on_close_raises = False
    
    def handle(self):
        for line in self.rfile:
            should_stop = self._notify_line(line)
            if should_stop or self.stop_requested:
                return
        self._notify_line(False)
        

if __name__ == '__main__':
        
    class Handler(AsyncLineRequestHandler):
        def raw_line_available(self, line):
            if line[0:4] == 'echo':
                self.wfile.write(line[4:].strip())
            elif line[0:4] == 'quit':
                return True
            elif line[0:4] == 'stop':
                self.request_stop()
                
        def gen_line_available(self):
            line = yield
            abort = False
            while line:
                if line[0:4] == 'echo':
                    self.wfile.write(line[4:].strip())
                elif line[0:4] == 'stop':
                    self.request_stop()
                elif line[0:4] == 'term':
                    self.server.initiate_shutdown()
                elif line[0:4] == 'quit':
                    return
                elif line[0:5] == 'raise':
                    raise ValueError(line[5:])
                elif line[0:4] == 'live':
                    abort = True
                line = yield
                if not line:
                    if abort:
                        print('Ignoring klill')
                        abort = False
                        self.stop_requested = False
                        line = yield
                    else:
                        print("I'm dying")
        
        line_available = gen_line_available
        #line_available = raw_line_available
        def finish(self):
            print('Finished')
    class ASS(StartStopMixin, SelectingUnixStreamServer):pass
    server = ASS('/tmp/sockets/socket',Handler,bind_and_activate=False)
    server.allow_reuse_address = True
    server.unlink_socket_on_start = True
    print(server)
    server.server_bind()
    print(server)
    server.server_activate()
    print(server)
    server.start()
    print(server)
    server.wait(60)
    print(server)
    print('Issued shutdown on Main')
    server.shutdown()
    print(server)
    
