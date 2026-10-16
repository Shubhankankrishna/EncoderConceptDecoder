"""Launch V9 without importing unrelated editable projects during site startup.

Global Anaconda .pth files remain untouched. The normal site hooks and installed
libraries are retained, except for editable source paths belonging to other
projects. Child Python processes inherit this launcher through sys.executable.
"""
import os
import site
import sys


def main():
    original = site.addpackage

    def addpackage(sitedir, name, known_paths):
        if name.startswith('__editable__.') or name == '_editable_impl_acg_v2.pth':
            return known_paths
        return original(sitedir, name, known_paths)

    site.addpackage = addpackage
    try:
        site.main()
    finally:
        site.addpackage = original

    import runpy
    from pathlib import Path

    sys.executable = str(Path(__file__).with_name('v9_python').resolve())
    args = sys.argv[1:]
    while args and args[0] in ('-B', '-S', '-s', '-u'):
        if args[0] == '-u':
            for stream in (sys.stdout, sys.stderr):
                stream.reconfigure(line_buffering=True, write_through=True)
        args.pop(0)
    if not args:
        raise SystemExit('V9 Python requires a script, -c command, or -m module')
    mode = args.pop(0)
    if mode == '-c':
        command = args.pop(0)
        sys.argv = ['-c', *args]
        sys.path[0] = os.getcwd()
        exec(compile(command, '<string>', 'exec'), {'__name__': '__main__'})
    elif mode == '-m':
        module = args.pop(0)
        sys.argv = [module, *args]
        sys.path[0] = os.getcwd()
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    else:
        if mode.startswith('-'):
            raise SystemExit('Unsupported V9 Python option: ' + mode)
        script = str(Path(mode).resolve())
        sys.argv = [script, *args]
        sys.path[0] = str(Path(script).parent)
        runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
