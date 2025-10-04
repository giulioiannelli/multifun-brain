# Frequently asked questions

## Is the package production ready?

The project is currently in **beta**. The APIs are mostly stable but may change
between minor releases. Pin the version for production workloads.

## Does the package ship real neuroimaging datasets?

No. Only synthetic generators are included. This keeps the distribution light
and avoids licensing issues. You can integrate your own datasets by loading them
with NetworkX or NumPy and feeding them into the provided analysis utilities.

## How do I cite multifun-brain?

```
Multifun-Brain Developers. (2024). multifun-brain (Version 0.2.0) [Computer software].
https://github.com/your-org/multifun-brain
```

## Where are the examples?

Explore the `notebooks/` directory for Jupyter notebooks that walk through
complete analysis pipelines. They can be executed after installing the `viz`
extra: `pip install "multifunbrain[viz]"`.

## I found a bug. What now?

1. Search the [issue tracker](https://github.com/your-org/multifun-brain/issues)
   to avoid duplicates.
2. If the issue is new, open a report using the provided template. Include
   Python version, operating system, and steps to reproduce the problem.
3. Pull requests with failing-test reproductions are highly appreciated.

## Which licence does the project use?

The project is released under the [MIT License](../LICENSE).
