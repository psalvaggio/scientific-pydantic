"""Generate API reference pages by walking __all__ in scientific_pydantic.

This script is run by mkdocs-gen-files at build time. It imports the package
and recursively follows __all__ through subpackages, creating one doc page per
discovered module/symbol and writing a SUMMARY.md for mkdocs-literate-nav.
"""

import importlib
import types
from pathlib import Path

import mkdocs_gen_files

PACKAGE = "scientific_pydantic"
API_DIR = Path("api")

nav = mkdocs_gen_files.Nav()


def process_module(module: types.ModuleType) -> None:  # noqa: C901, PLR0912
    """Recursively process a module, following subpackages found in __all__"""
    all_exports = getattr(module, "__all__", None)
    if all_exports is None:
        return

    dotted = module.__name__
    parts = dotted.split(".")

    # Determine doc path: packages get index.md, plain modules get <name>.md
    is_package = hasattr(module, "__path__")
    if is_package:
        doc_path = API_DIR / Path(*parts) / "index.md"
    else:
        doc_path = API_DIR / Path(*parts[:-1]) / f"{parts[-1]}.md"

    nav[("API Reference", *parts[1:])] = str(doc_path)

    # Separate subpackages from plain exported symbols so we don't
    # double-document things that will get their own pages.
    subpackage_names = set()
    for name in all_exports:
        obj = getattr(module, name, None)
        if isinstance(obj, types.ModuleType) and hasattr(obj, "__path__"):
            subpackage_names.add(name)

    exported_symbols = [n for n in all_exports if n not in subpackage_names]

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# `{parts[-1]}`\n\n")
        fd.write(f"```python\nimport {dotted}\n```\n\n")
        if exported_symbols:
            fd.write(f"::: {dotted}\n")
            fd.write("    options:\n")
            fd.write(f"      members: {exported_symbols}\n")
        else:
            fd.write("*See subpackages below.*\n")

    # Point the "edit this page" link at the actual source file
    mkdocs_gen_files.set_edit_path(doc_path, module.__file__)

    # Recurse into any subpackages or submodules listed in __all__
    for name in all_exports:
        obj = getattr(module, name, None)
        if isinstance(obj, types.ModuleType):
            if hasattr(obj, "__path__"):
                # Already a fully imported subpackage
                process_module(obj)
            else:
                # Plain submodule — import it explicitly and recurse if it has __all__
                full_name = f"{dotted}.{name}"
                try:
                    submod = importlib.import_module(full_name)
                    if hasattr(submod, "__all__"):
                        process_module(submod)
                except ImportError:
                    pass


root_module = importlib.import_module(PACKAGE)
process_module(root_module)

# Write SUMMARY.md consumed by mkdocs-literate-nav
with mkdocs_gen_files.open("SUMMARY.md", "a") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
