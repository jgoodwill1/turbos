def time_average(
    run,
    slices,
    variables,
    transform=None,
):
    slices = list(slices)

    if not slices:
        raise ValueError("slices cannot be empty")

    sums = None
    template = None

    for index in slices:
        print(index)
        ds = run.load_xarray(
            index,
            variables=variables,
        )

        if transform is not None:
            ds = transform(ds)

        if sums is None:
            template = ds
            sums = {
                name: ds[name].values.copy()
                for name in ds.data_vars
            }
        else:
            for name in ds.data_vars:
                sums[name] += ds[name].values

    n = len(slices)

    out = template.copy(deep=False)

    for name in out.data_vars:
        out[name].data = sums[name] / n

    out.attrs["n_averaged"] = n

    return out