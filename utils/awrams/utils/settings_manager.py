
import os

HOME = os.path.expanduser('~')

def package_get_settings(package):
    """
    Helper for importing package level settings from the user's home directory
    Should only be used by awrams packages which override their default settings.

    For user level settings import use get_settings (see below)

    Allows a particular package (eg `awrams.visualisation`) to import user specific
    settings from either `~/.awrams/visualisation.py` or `~/.awrams/settings.py`
    using package_get_settings('visualisation')

    Intended use is from a package specific `settings.py` of equivalent that looks like

    SOME_SETTING=#default value
    SOME_OTHER_SETTING=#default value

    # end of file, load and override defaults with anything from the user's
    # own settings file
    from .settings_manager import package_get_settings as _get_settings
    exec(_get_settings('package_name'))
    """

    fn = os.path.join(HOME,'.awrams',package+'.py')
    if not os.path.exists(fn):
        fn = os.path.join(HOME,'.awrams','settings.py')

    if os.path.exists(fn):
        return compile(open(fn).read(), fn, 'exec')
    else:
        return compile('','<string>','exec')

def get_all_settings():
    from awrams.utils.metatypes import ObjectDict
    import importlib.machinery
    import types

    HOME = os.path.expanduser('~')

    sdict = ObjectDict()

    import glob
    local_setting_files = glob.glob(os.path.join(HOME,'.awrams/*.py'))
    for f in local_setting_files:
        submod = os.path.split(f)[-1].split('.')[0]
        modname = 'awrams.{submod}.settings'.format(**locals())
        try:
            mod = importlib.import_module(modname)
            sdict[submod] = mod
        except:
            loader = importlib.machinery.SourceFileLoader(submod,f)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            sdict[submod] = mod

    return sdict


def get_settings(package):
    import importlib.machinery
    import types

    HOME = os.path.expanduser('~')

    modname = 'awrams.{package}.settings'.format(**locals())
    try:
        m = importlib.import_module(modname)
        return m
    except:
        try:
            f = os.path.join(HOME,'.awrams/{package}.py'.format(**locals()))
            loader = importlib.machinery.SourceFileLoader(package,f)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            return mod
        except:
            raise Exception("No settings found for package {package}".format(**locals()))
