from invoke import task


@task
def test(c):
    c.run("python -m unittest")


@task
def upload(c):
    c.run("rm -rf dist")
    c.run("python -m build")
    c.run("scp dist/* inca02.pri.bms.com:/web/msdpypi/packages/")
