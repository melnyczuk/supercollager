import logging
from sys import stdout

logging.disable(logging.INFO)


def describe(test):
    def it(*args):
        class_name = args[0].__class__.__name__.replace("TestCase", "")
        function_name = test.__name__.replace("test_", "")
        stdout.write(f"\n\n{class_name}: {function_name}\n")
        test(*args)

    return it


def each(each):
    def it(test):
        test_name = test.__name__.replace("_", " ")
        each_str = "\n    ".join([str(e) for e in each])
        stdout.write(f"  it {test_name} for each in \n    {each_str} \n")
        for e in each:
            try:
                test(e)
            except Exception as ex:
                stdout.write(f"  fail: {test_name} for {each_str} \n")
                raise ex

    return it


def it(test):
    test_name = test.__name__.replace("_", " ")
    stdout.write(f"  it {test_name} \n")
    try:
        test()
    except Exception as ex:
        stdout.write(f"  fail: {test_name} \n")
        raise ex
