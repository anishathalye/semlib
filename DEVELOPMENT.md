# Development

Semlib uses the [Hatch] project manager ([installation instructions][hatch-install]).

Hatch automatically manages dependencies and runs testing, type checking, and other operations in isolated [environments][hatch-environments].

[Hatch]: https://hatch.pypa.io/
[hatch-install]: https://hatch.pypa.io/latest/install/
[hatch-environments]: https://hatch.pypa.io/latest/environment/

## Testing

You can run the tests on your local machine with:

```bash
hatch test
```

The [`test` command][hatch-test] supports options such as `-c` for measuring test coverage, `-a` for testing with a matrix of Python versions, and appending an argument like `hatch test tests/semlib/test_prompt.py::test_prompt` for running a single test.

[hatch-test]: https://hatch.pypa.io/latest/tutorials/testing/overview/

## Type checking

You can run the [mypy static type checker][mypy] with:

```bash
hatch run types:check
```

You can run the type checker on the `.ipynb` notebooks in the documentation with:

```bash
hatch run types:check-docs
```

[mypy]: https://mypy-lang.org/

## Formatting and linting

You can run the [Ruff][ruff] formatter and linter with:

```bash
hatch fmt
```

This will automatically make [safe fixes][fix-safety] to your code. If you want to only check your files without making modifications, run `hatch fmt --check`.

[ruff]: https://github.com/astral-sh/ruff
[fix-safety]: https://docs.astral.sh/ruff/linter/#fix-safety

## Docs

You can preview the [MkDocs][mkdocs] site locally at <http://localhost:8000/> with:

```bash
hatch run docs:serve
```

This will live-reload the site as you make changes. You can run `hatch run docs:build` to build the compiled site into the `site/` directory.

[mkdocs]: https://www.mkdocs.org/

## Packaging

You can use [`hatch build`][hatch-build] to create build artifacts, a [source distribution ("sdist")][sdist] and a [built distribution ("wheel")][bdist].

You can use [`hatch publish`][hatch-publish] to publish build artifacts to [PyPI][pypi].

[hatch-build]: https://hatch.pypa.io/latest/build/
[sdist]: https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist
[bdist]: https://packaging.python.org/en/latest/glossary/#term-Built-Distribution
[hatch-publish]: https://hatch.pypa.io/latest/publish/
[pypi]: https://pypi.org/

## Continuous integration

Testing, type checking, and formatting/linting is [checked in CI][ci].

[ci]: .github/workflows/ci.yml
