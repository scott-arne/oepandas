from invoke import task


@task
def test(c):
    c.run("python -m pytest")


@task
def build(c):
    """Build the package for distribution"""
    c.run("rm -rf dist")
    c.run("python -m build")
