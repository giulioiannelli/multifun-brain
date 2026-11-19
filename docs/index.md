# multifun-brain documentation

Welcome to the **multifun-brain** documentation hub. The project provides a
portable toolkit for building and analysing hierarchical modular brain networks.
This site organises background information, environment setup, tutorials, and API
references. All pages are Markdown so they render nicely on GitHub and can be
served as a website with MkDocs.

## Table of contents

1. [Installation](installation.md)
2. [Usage guide](usage.md)
3. [API reference](api_reference.md)
4. [Development guide](development.md)
5. [Frequently asked questions](faq.md)

## At a glance

- **Package name:** `multifunbrain`
- **Python versions:** 3.9 – 3.12
- **License:** MIT
- **Source code:** <https://github.com/giulioiannelli/multifun-brain>
- **Issue tracker:** <https://github.com/giulioiannelli/multifun-brain/issues>

## Building the documentation site

Install the documentation extras and launch MkDocs:

```bash
pip install "multifunbrain[docs]"
mkdocs serve
```

MkDocs automatically watches for changes in the `docs/` directory. When you are
happy with the content run `mkdocs build` to generate a static site in
`site/`.
