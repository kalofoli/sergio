'''
Created on Jun 21, 2018

@author: janis
'''
from typing import Callable, List, NamedTuple, Any, Dict, Sequence, Tuple, cast,\
    Union
import inspect
import re

import functools

# pylint: disable= bad-mcs-classmethod-argument, too-few-public-methods

from .summaries import SummarisableFromFields
from types import SimpleNamespace


class ParsedArgument(NamedTuple):
    key:str
    value:Any
    quote:str
    text:str


try:
    from recordclass import RecordClass
    class ProductBundle(RecordClass):
        factory_object: 'FactoryBase'
        method_object: 'FactoryMethod'
        functor: Callable
        name: str
        args: Dict[str, Any]
        digest:str
        product: Any
        
        def produce(self, obj=None):
            product = self.functor(obj, **self.args) if obj is not None else self.functor(**self.args)
            self.product = product
            return product
        
        @property
        def __summary_name__(self):
            return self.name
    
    
    ProductBundle.__bases__ = ProductBundle.__bases__ + (SummarisableFromFields,)
    ProductBundle.__summary_fields__ = ('name','args','digest','product')
        
except ImportError:
    class ProductBundle(SimpleNamespace, SummarisableFromFields):
        __summary_fields__ = ('name','args','digest','product')
        def __init__(self, factory_object: 'FactoryBase', method_object: 'FactoryMethod',
                     functor: Callable, name: str, args: Dict[str, Any], digest:str, product: Any):
            self.factory_object: 'FactoryBase' = factory_object
            self.method_object: 'FactoryMethod' = method_object
            self.functor: Callable = functor
            self.name: str = name
            self.args: Dict[str, Any] = args
            self.digest:str = digest
            self.product: Any = product
        
        def produce(self, obj=None):
            product = self.functor(obj, **self.args) if obj is not None else self.functor(**self.args)
            self.product = product
            return product
        
        @property
        def __summary_name__(self):
            return self.name
    


class TypeResolver:

    def resolve(self, cls, txt):
        raise NotImplementedError(f'Use one of the subclasses.')

class DefaultTypeResolver(TypeResolver):
    BOOL_TRUE = {'true', '1', 'on', 'yes'}
    BOOL_FALSE = {'false', '0', 'off', 'no'}

    def resolve(self, cls, txt):
        import enum
        factory = cls
        if issubclass(cls, bool):
            if txt.lower() in self.BOOL_TRUE:
                return True
            elif txt.lower() in self.BOOL_FALSE:
                return False
            else:
                raise ValueError(f'Could not parse a boolean from {txt}.')
        elif issubclass(cls, enum.Enum):
            from colito.resolvers import make_enum_resolver
            factory = make_enum_resolver(cls).resolve
        return factory(txt)


DEFAULT_TYPE_RESOLVER = DefaultTypeResolver()


class DictTypeResolver(TypeResolver):

    def __init__(self, types, default=DEFAULT_TYPE_RESOLVER):
        self._types = types
        self._default = default
    
    def resolve(self, cls, txt):
        if cls in self._types:
            resolver = self._types[cls]
        elif cls.__name__ in self._types:
            resolver = self._types[cls.__name__]
        else:
            if self._default is None:
                raise KeyError(f'Type {cls.__name__} not specified in dictionary and no default resolver provided.')
            else:
                resolver = lambda txt: self._default(cls, txt)
        return resolver(txt)

    
class FactoryMethod:
    
    def __init__(self, method, name=None):
        self.method = method
        self.is_static = isinstance(method, staticmethod)
        self.is_class = isinstance(method, classmethod)
        self.functor = method.__func__ if self.is_static or self.is_class else method
        if name is None:
            name = self.functor.__name__
        self.name = name
        parameters = inspect.signature(self.functor).parameters  # Drop the self/cls parameter
        self.name2argument = {argument.name: argument for argument in parameters.values()}
        self.name2index = {argument.name: index for index, argument in enumerate(parameters.values())}
        self.parameters = list(parameters.values())
        
    def assign_parameters_from_parsed_arguments(self, arguments:Sequence[ParsedArgument], obj=None, type_resolver:TypeResolver=DEFAULT_TYPE_RESOLVER, ignore_extra_arguments:bool=False) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        missing = object()
        if len(self.parameters) < len(arguments) and not ignore_extra_arguments:
            raise TypeError(f'While calling factory {self.name}: Provided {len(arguments)} arguments but at most {len(self.parameters)} are expected.')
        
        # Sort arguments based on their keys, if specified
        keyword_argument_preceded = False
        offset = 1 if not self.is_static else 0
        parameters = self.parameters[offset:]
        parameter_values:List[Union[ParsedArgument, object]] = [missing] * len(parameters)
        for index, argument in enumerate(arguments):
            if argument.key is None:
                if keyword_argument_preceded:
                    raise TypeError(f'Non-keyword argument {argument} at {index} cannot follow keyword argument.')
                parameter_values[index] = argument
            else:
                argument = cast(ParsedArgument, argument)
                if not keyword_argument_preceded:
                    keyword_argument_preceded = True
                index_key = self.name2index[argument.key] - offset
                if parameter_values[index_key] is not missing:
                    raise TypeError(f'Double argument assignment for argument {index_key}')
                parameter_values[index_key] = argument
                
        # Assign defaults
        args = [] if self.is_static else [obj]
        kwargs = {}
        args_dct:Dict[str, Any] = {}
        for index, assignment, parameter in zip(range(len(parameter_values)), parameter_values, parameters):
            if assignment is missing:
                if parameter.default is inspect.Signature.empty:
                    raise TypeError(f'Missing value for argument {parameter.name} at position {index}.')
                value = parameter.default
                has_key = True 
            else:
                assignment = cast(ParsedArgument, assignment)
                if parameter.annotation is not inspect.Signature.empty:
                    value = type_resolver.resolve(parameter.annotation, assignment.text)
                else:
                    value = assignment.text
                has_key = assignment.key is not None
                assert not has_key or parameter.name == assignment.key, f'INTERNAL: Keys mismatch. Parameter was {parameter.name} but key in assignment was {assignment.key}.'
            if has_key:
                kwargs[parameter.name] = value
            else:
                args.append(value)
            args_dct[parameter.name] = value
        return args, kwargs, args_dct
    
    def __call__(self, obj, *args, **kwargs):
        return self.functor(obj, *args, **kwargs)
    
    def __repr__(self):
        type_txt = 'class ' if self.is_class else ('static ' if self.is_static else '')
        return f'<{self.__class__.__name__} mapping "{self.name}" to {type_txt}method "{self.functor.__name__}">'
    
    def make_caller(self):
        '''For use from the metaclass: make a function calling the sub-method directly'''
        return self.method
        
'''Decorators'''
def factorymethod(name=None):

    def decorator(method):
        nonlocal name
        return FactoryMethod(method=method, name=name)

    return decorator

def factorygetter(type_resolver:TypeResolver=DEFAULT_TYPE_RESOLVER, member=None,
                  parameter_separator=',', name_separator=':', key_value_separator='=',
                  allow_extra_arguments=False, default=None):
    
    def decorator(method):
        is_class = isinstance(method, classmethod)
        func = method.__func__ if is_class else method
        getter = FactoryGetter(type_resolver=type_resolver, member=member,classmethod=is_class,
                  parameter_separator=parameter_separator, name_separator=name_separator, key_value_separator=key_value_separator,
                  allow_extra_arguments=allow_extra_arguments, default=default)
        def wrapper(obj, text, *args, **kwargs):
            result = getter(obj, text)
            return func(obj, result, *args, **kwargs)
        functools.update_wrapper(wrapper, func)
        wrapper_out = classmethod(wrapper) if is_class else wrapper
        return wrapper_out
    return decorator

class FactoryGetter:
    
    def __init__(self, type_resolver:TypeResolver=DEFAULT_TYPE_RESOLVER, member=None,
                 parameter_separator=',', name_separator=':', key_value_separator='=',
                 allow_extra_arguments=False, default=None, classmethod=False) -> None:
        '''
        @param member: the member of the ProductBundle that this getter returns. None returns all of the bundle.
        '''
        self.type_resolver:TypeResolver = type_resolver
        if member is not None and not hasattr(ProductBundle, member):
            raise KeyError(f'Invalid member: "{member}". Choose one of {",".join(ProductBundle._fields)}')
        self.member = member
        self.name_separator = name_separator
        self.parameter_separator = parameter_separator
        self.key_value_separator = key_value_separator
        self.allow_extra_arguments = allow_extra_arguments
        self.default = default
        self.classmethod = classmethod
    
    _match_sq = r"'(?P<val_sq>[^']*(\\'[^']*)*)(?<!\\)'"
    _match_dq = r'"(?P<val_dq>[^"]*(\\"[^"]*)*)(?<!\\)"'
    _match_id = r'(?P<val>[^"\'][^,\\]*)'
    _match_key = r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)'
    
    def parse_arguments_from_string(self, text):
        parts = text.split(self.name_separator)
        rex = re.compile(fr'(?:{self._match_key}\s*{self.key_value_separator})?\s*(?:{self._match_dq}|{self._match_sq}|{self._match_id})\s*(?P<rest>.*?)$')
        
        parsed_arguments:List[ParsedArgument] = []
        if len(parts) > 1:
            name, str_args = parts
            index = len(name) + 1
            rest = str_args
            while True:
                match = rex.match(rest)
                if match is None:
                    raise ValueError(f'Error while parsing arguments after "{text[0:index]}[{text[index+1:]}]".')
                dct = match.groupdict()
                if dct['val_dq'] is not None:
                    value = dct['val_dq']
                    quote = '"'
                elif dct['val_sq'] is not None:
                    value = dct['val_sq']
                    quote = "'"
                else:
                    value = dct['val']
                    quote = None
                argument = ParsedArgument(key=dct['key'], value=None, quote=quote, text=value)
                parsed_arguments.append(argument)
                index += len(rest) - len(dct['rest'])
                rest = dct['rest']
                if not rest:
                    break
                if rest[0] != self.parameter_separator:
                    raise ValueError(f'Fail to read up to separator near "{text[:index]}[{text[index]}]{text[index+1:]}".')
                rest = rest[1:]
                index += 1
        else:
            name = parts[0]
    
        return name, parsed_arguments
    
    def __call__(self, obj, text,*args, **kwargs):
        return self.get(obj, text,*args, **kwargs)
    
    def make_getter(self):

        def getter(obj, text, *args, **kwargs):
            return self.__call__(obj, text,*args, **kwargs)

        return classmethod(getter) if self.classmethod else getter
    
    def get(self, obj, what=None, *args,**kwargs):
        if what is None:
            if self.default is not None:
                what = self.default
            else:
                raise TypeError(f'A description is required unless a default is specified.') 
        if isinstance(what, ProductBundle):
            bundle = cast(ProductBundle, what)
        elif isinstance(what, str):
            text = cast(str, what)
            bundle = self.get_bundle_from_text(obj, text, *args, **kwargs)
        else:
            raise TypeError(f'Unknown type {what.__class__.__name__} for object {what}. You can specify either text or a ProductBundle with the same class.')
        result = bundle if self.member is None else getattr(bundle, self.member)
        return result
    
    def get_bundle_from_text(self, obj, text, produce:bool=True) -> ProductBundle:
        cls = obj if self.classmethod else obj.__class__
        factories = cls._factories
        factory_name, arguments = self.parse_arguments_from_string(text)
        try:
            factory_method:FactoryMethod = factories[factory_name]
        except KeyError:
            raise KeyError(f'No method {factory_name} in class {cls.__name__}. Available factories are: [{cls._factory_description}].')
        if factory_method.is_class:
            fn_obj = cls
        elif factory_method.is_static:
            fn_obj = None
        else:
            if self.classmethod:
                raise TypeError(f'Invoking a member method requires an instance.')
            fn_obj = obj
        args, kwargs, args_merged = factory_method.assign_parameters_from_parsed_arguments(arguments, obj=fn_obj, type_resolver=self.type_resolver, ignore_extra_arguments=self.allow_extra_arguments)
        product = factory_method(*args, **kwargs) if produce else None
        bundle = self.make_product(product=product, args_merged=args_merged, factory_method=factory_method, obj=obj)
        return bundle
    
    def make_product(self, product, args_merged, factory_method, obj):
        args_txt = self.parameter_separator.join(f'{key}{self.key_value_separator}{value!r}' for key, value in args_merged.items())
        name = factory_method.name
        digest_txt = f'{name}{self.name_separator}{args_txt}' if args_txt else name
        bundle = ProductBundle(factory_object=obj,method_object=factory_method, functor=factory_method.functor, name=factory_method.name, args=args_merged, digest=digest_txt, product=product)
        return bundle

class FactoryDescription:
    '''Creates a description of the options in the factory.
    
    Can be adapted to one's needs by appropriate subclassing.
    '''
    def __init__(self, choice_separator = ', '):
        self._choice_separator = choice_separator
        
    def factory2str(self, factory:FactoryMethod) -> str:
        func=factory.functor
        return self._choice_separator.join(map(str, tuple(inspect.signature(func).parameters.values())[1:])) 
        
    def describe(self, factories:Sequence[FactoryMethod]) -> str:
        return ','.join(f'{factory.name}({self.factory2str(factory)})' for factory in factories)

class FactoryMeta(type):
    
    def __new__(cls, cls_name, bases, dct):
        # Compute factories
        factories = {val.name:val for key, val in dct.items() if isinstance(val, FactoryMethod)}
        dct['_factories'] = factories
        dct['_factory_description'] = FactoryDescription().describe(factories.values())
        for key, value in dct.items():
            if isinstance(value, FactoryGetter):
                value = cast(FactoryGetter, value)
                dct[key] = value.make_getter()
            elif isinstance(value, FactoryDescription):
                value = cast(FactoryDescription, value)
                dct[key] = value.describe(factories.values())
            elif isinstance(value, FactoryMethod):
                value = cast(FactoryMethod, value)
                dct[key] = value.make_caller()
        cls_new = super(FactoryMeta, cls).__new__(cls, cls_name, bases, dct)
        return cls_new
    

class FactoryBase(metaclass=FactoryMeta):
    _factories:Dict[str, FactoryMethod]
    _factory_description:str


class NoConversionException(RuntimeError):
    pass

def resolve_arguments(method, args, kwargs, resolver=DEFAULT_TYPE_RESOLVER, handler=None, kwargs_resolver=None, head_args=(None,)):
    """ Parse arguments to a function using annotations and inspection heuristics.
    
        :param allow_unbound_kwargs: Raise an error if a provided kwarg cannot be bounded. Defaults to False.
        :type allow_unbound_kwargs: bool  
        :param handler: a function that handles individual named parameters.
            Has the signature: handler(name, value, parameter)
            If the default conversion is to be used, it must raise a NoConversionException.
    """
    sig = inspect.signature(method)
    try:
        if kwargs_resolver is None:
            bound_args = sig.bind(*head_args, *args, **kwargs)
        else:
            bound_args_partial = sig.bind_partial(*head_args, *args, **kwargs)
            kwargs_unresolved = {}
            for key, par in sig.parameters.items():
                if key in bound_args_partial.arguments: continue
                if par.kind in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}: continue
                if par.default is not inspect.Parameter.empty: continue
                kwargs_unresolved[key] = par
            kwargs_resolved = kwargs_resolver(kwargs_unresolved, signature=sig, bound_args=bound_args_partial) 
            bound_args = sig.bind(*head_args, *args, **kwargs, **kwargs_resolved)
        parameters = bound_args.signature.parameters
        arguments = bound_args.arguments
    except TypeError as e:
        text = describe_resolvable_arguments(method, exclude={'self',}, sep='\n')
        raise TypeError(f"Could not parse function {method} because of {e}. Args: \n{text}") from e
    for name,val_in in arguments.items():
        parameter = parameters[name]
        try:
            if handler is not None:
                val = handler(name, val_in, parameter)
                arguments[name] = val
                continue
        except NoConversionException:
            pass
        if parameter.annotation is not inspect.Parameter.empty:
            parameter_type = parameter.annotation
            if not isinstance(val_in, parameter_type):
                try:
                    val = resolver.resolve(parameter_type, val_in)
                    arguments[name] = val
                except Exception as e:
                    raise TypeError(f'While parsing parameter {name} from {val_in} into {parameter_type}') from e
    bound_args.apply_defaults()
    return bound_args.args, bound_args.kwargs


def default_string_argument_handler(name, value, parameter):
    raise NoConversionException()

def make_string_constructor(class_resolver):
    @classmethod
    def string_constructor(cls, name, *args, **kwargs):
        kernel_cls = class_resolver.get_class_from_tag(name)
        init_prototype = kernel_cls._init_args if hasattr(kernel_cls, '_init_args') else kernel_cls.__init__
        handler = kernel_cls.parse_string_argument if hasattr(kernel_cls,'parse_string_argument') else None
        args_p, kwargs_p = resolve_arguments(init_prototype, args, kwargs, handler=handler)
        kernel = kernel_cls(*args_p[1:], **kwargs_p)
        return kernel
    return string_constructor


rex_tag = re.compile('(?P<cls>[-a-zA-Z0-9_]+)(\((?P<args>[^()]+)\))?(\{(?P<kwargs>[^{}]+)\})?')
def parse_constructor_string(s, tag='constructor'):
    from ast import literal_eval
    m = rex_tag.match(s)
    if m is None:
        raise ValueError(f'Could not parse a valid {tag} from "{s}". The correct format is: name(v1,v2){{k2:v2,...}}')
    cls_name = m.group('cls')
    sargs = m.group('args')
    if sargs is None:
        args = []
    else:
        try:
            args = literal_eval(f'[{sargs}]')
        except RuntimeError as e:
            raise ValueError(f'Could not parse arguments from "{sargs}".') from e
    skwargs = m.group('kwargs')
    if skwargs is None:
        kwargs = {}
    else:
        try:
            kwargs = literal_eval(f'{{{skwargs}}}')
        except RuntimeError as e:
            raise ValueError(f'Could not parse arguments from "{skwargs}".') from e
    return cls_name, args, kwargs
    
    
def describe_resolvable_arguments(what, exclude={'self',}, sep=None):
    """ Create a description of resolvable arguments."""
    if isinstance(what, type):
        cls = what
        method = cls._init_args if hasattr(cls, '_init_args') else cls.__init__
    else:
        method = what
    sig = inspect.signature(method)
    arg_texts = [str(parameter) for name,parameter in sig.parameters.items()
                 if name not in exclude]
    if sep is not None:
        res = sep.join(arg_texts)
    else:
        res = arg_texts
    return res
    
