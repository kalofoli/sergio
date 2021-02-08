'''
Created on Feb 1, 2021

@author: janis
'''
from datetime import datetime
import os

from .summarisable import SummarisableAsDict
from .logging import getModuleLogger

from . import property_eval_once

log = getModuleLogger(__name__)

class RuntimeEnvironment(SummarisableAsDict):
    _fields = ('date', 'hostname', 'username', 'pid', 'cwd', 'cpu_count', 'git_version')
    
    def __init__(self):
        self.date = datetime.DateTime()

    @property
    def pid(self):
        return os.getpid()

    @property
    def cwd(self):    
        return os.getcwd()

    @property_eval_once
    @staticmethod
    def cpu_count():
        return os.cpu_count()
    
    @property_eval_once
    @staticmethod
    def username():
        try:
            import pwd
            return pwd.getpwuid(os.getuid()).pw_name
        except:
            return 'unknown'
    
    @property_eval_once
    @staticmethod
    def hostname():
        try:
            import socket
            return socket.gethostname()
        except:
            return 'unknown'
    
    @property_eval_once
    @staticmethod
    def git_version():
        try:
            import subprocess
            cwd = os.path.dirname(os.path.realpath(__file__))
            sp = subprocess.Popen('git describe --tags --always --dirty --long'.split(' '),
                                  cwd=cwd,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            txt_out, txt_err = sp.communicate()
            if sp.returncode != 0:
                log.warning(f'Failed to get git version: {txt_err.decode("latin")}')
                return f'git-return-{sp.returncode}'
            else:
                return txt_out.decode('latin').strip()
        except Exception as e:
            return f'error-{e[0]}'
    
    def summary_dict(self, options):
        return self.summary_from_fields(self._fields)

