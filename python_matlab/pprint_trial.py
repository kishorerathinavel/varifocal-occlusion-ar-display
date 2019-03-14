import pprint
stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
stuff.insert(0, stuff[:])
print(type(pprint))
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(stuff)

