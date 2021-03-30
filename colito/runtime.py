'''
Created on Feb 1, 2021

@author: janis
'''
from datetime import datetime
import os

from .logging import getModuleLogger

from . import property_eval_once
from .summaries import SummarisableFromFields

log = getModuleLogger(__name__)

class RuntimeEnvironment(SummarisableFromFields):
    __summary_fields__ = ('date', 'hostname', 'username', 'pid', 'cwd', 'cpu_count', 'git_version')
    __summary_conversions__ = {'date':str}
    def __init__(self):
        self.date = datetime.now()

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

