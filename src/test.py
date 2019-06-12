"""
Test for package import
"""


try:
    import ref.bert.modeling
except:
    print('import ref.bert.modeling does not work')

try:
    from ref.bert import optimization
except:
    print('from ref.bert import optimization does not work')