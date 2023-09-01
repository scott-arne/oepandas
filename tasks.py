from invoke import task


@task
def test(c):
    c.run("python -m unittest")


@task
def msd_upload(c):
    c.run("rm -rf dist")
    c.run("python setup.py sdist --formats=gztar")
    c.run("scp dist/*.tar.gz inca02.pri.bms.com:/web/msdpypi/packages/")
