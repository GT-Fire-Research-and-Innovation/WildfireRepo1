"""FAH: the raw upstream data behind every fire of the wildfire spread dataset, one channel at a time.

Layers, each importing only from the ones below it (tests/test_architecture.py checks this):

    fah.__main__   command line
    commands       one class per command: run a channel, status, links, extents, repack, check
    channels       one class per channel; hub   Hugging Face repo, manifests, batches, fire catalog
    sources        one client per upstream service; storage   file formats written to the repo
    core           fires, geometry, run context, results, secrets, errors

Run it as `python -m fah --help`.
"""
