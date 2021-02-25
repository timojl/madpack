from os.path import basename, splitext
import re
import inspect


def get_args_string(f):

    args = []
    sig = inspect.signature(f).parameters
    for name, param in sig.items():
        if param.default != inspect._empty:
            arg = f'<span class="arg_optional">{name}={str(param.default)}</span>'
        else:
            arg = f'<span class="arg">{name}</span>'

        args += [arg]

    return ', '.join(args)


class ListHandler:
    def __init__(self, name): 
        self.args = dict()
        self.last = None
        self.name = name

    def add_kv(self, k, v):
        self.args[k] = v
        self.last = k

    def add_line(self, line):
        self.args[self.last] += line

    def get_html(self):
        out = f'<b>{self.name}</b><ul>'
        for k, v in self.args.items():
            out += f'<li><code>{k}</code>:&nbsp;{v}</li>'
        out += '</ul>'
        return out


class TextHandler:
    def __init__(self, name): 
        self.lines = []
        self.name = name

    def add_line(self, line):
        if len(line.strip()) > 0:
            self.lines += [line.strip()]

    def get_html(self):
        return '<p style="margin-top: 0.1em; margin-left: 1em">' + '<br />'.join(self.lines) + '</p>'


class CodeHandler(TextHandler):
    def add_line(self, line):
        self.lines += [re.sub('>>>?', '', line)]

    def get_html(self):
        return f'<br /><b>{self.name}</b><br /><code>' + '<br />'.join(self.lines) + '</code>'


def parse_docstring(function):

    if function.__doc__ is None:
        return {}

    tree = dict()
    current = None
    handlers = {
        'Args': ListHandler,
        'Arguments': ListHandler,
        'Parameters': ListHandler,
        'Examples': CodeHandler,
        'Example': CodeHandler,
        'Returns': TextHandler,
    }

    tree[current] = TextHandler('')

    for line in function.__doc__.split('\n'):
        
        m = re.match(r'\s*(\w*):\s*([\s\w]*)', line)
        if m and m.group(2) == '':
            current = m.group(1)
            tree[current] = handlers[current](m.group(1))
        elif m and current is not None and hasattr(tree[current], 'add_kv'):
            tree[current].add_kv(m.group(1), m.group(2))
        else:
            tree[current].add_line(line)

    return tree


def html_docstring(function):
    html_args = get_args_string(function)
    html = f'<div style="font-size: medium; font-family: monospace"><b>{function.__name__}</b>({html_args})</div>'
    tree = parse_docstring(function)
    html += ''.join(v.get_html() for k, v in tree.items())
    return html


def html_docstring_class(cls, exclude=()):
    html = html_docstring(cls)
    methods = vars(cls)
    methods = [m for m in methods if m not in ('__module__', '__doc__') + exclude
               and callable(getattr(cls, m))]

    html += "<ul>"
    for method in methods:
        html += '<li>' + html_docstring(getattr(cls, method)) + '</li>'
    html += '</ul>'
    return html


def apply_marks(text, marks):
    for label, content in marks.items():
        text = text.replace('<<' + label + '>>', content)
    return text


def run_notebook(filename, execute=True):
    import nbformat
    from pygments.formatters.html import HtmlFormatter
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import ExecutePreprocessor

    from madpack import __version__
    version = str(__version__)

    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)

    if execute:
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'doctests/'}})

    with open(filename, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    html_exporter = HTMLExporter()
    html_exporter.template_file = 'basic'
    (body, resources) = html_exporter.from_notebook_node(nb)

    filename_body = splitext(basename(filename))[0]
    pygments_css = HtmlFormatter().get_style_defs('.highlight')
    template = open('doctests/_templates/base.html').read()
    template_marks = dict(
        pygments_css=pygments_css,
        content=body,
        version=version,
    )
    main_html = apply_marks(template, template_marks)

    open(filename[:filename.rindex('.')] + '.html', 'w').write(main_html)

    return filename_body
