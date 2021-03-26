
import enum

from colito.files import EnumFileManager as _EnumFileManager


class FileKinds(enum.Enum):
    DATA = enum.auto()
    DATA_MORRIS = enum.auto()
    LOG = enum.auto()
    DEFAULT = enum.auto()
    
class FileManager(_EnumFileManager):
    Kinds = FileKinds
    __default_kind__ = FileKinds.DEFAULT
