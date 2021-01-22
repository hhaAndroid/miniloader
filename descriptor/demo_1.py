def demo_1():
    class Age(object):
        def __init__(self, age):
            self.val = age

        # # 如果不实现，则cls.x会返回Age实例对象
        def __get__(self, obj, objtype):
            print('Age.__get__')
            return self.val

    class MyClass(object):
        x = Age(12)  # x 是类属性

        def __init__(self):
            self.y = Age(10)  # y 是实例属性
            self.z = 5

    cls = MyClass()
    print(cls.x)
    print(cls.y)
    print(cls.z)


def demo_2():
    class MyClass(object):
        x = 4

        def __init__(self):
            self.y = 5

        def __getattribute__(self, item):
            print('MyClass.__getattribute__')
            return super(MyClass, self).__getattribute__(item)
            # return object.__getattribute__(self, item)

        def __getattr__(self, item):
            print('MyClass.__getattr__')
            raise AttributeError(f'{type(self).__name__} object had not attribute \'{item}\'')

        def __setattr__(self, key, value):
            print('MyClass.__setattr__')
            # self.key = value # 错误
            self.__dict__[key] = value

    print('开始实例化')
    cls = MyClass()
    print('--类属性访问--')
    print(MyClass.x)
    print('--类属性赋值--')
    MyClass.x = 1
    print('--实例属性访问--')
    print(cls.y)
    print('--实例属性赋值--')
    cls.y = 1
    print('--访问不存在的属性--')
    print(MyClass.k)
    print(cls.k)


def demo_3():
    class Age(object):
        def __init__(self, age):
            self.val = age

        def __getattribute__(self, item):
            print('Age.__getattribute__')
            return super(Age, self).__getattribute__(item)

        def __get__(self, obj, objtype):
            print(obj, objtype)
            print('Age.__get__')
            return self.val

    class MyClass(object):
        x = Age(12)  # x 是类属性

    # print(MyClass.__dict__)
    print(MyClass.x)
    print(MyClass().x)


def demo_4():
    class Age(object):
        def __init__(self, age):
            self.val = age

        # def __getattribute__(self, item):
        #     print('Age.__getattribute__')
        #     return super(Age, self).__getattribute__(item)

        def __get__(self, obj, objtype):
            print('Age.__get__')
            return self.val

        # def __set__(self, instance, value):
        #     pass

    class MyClass(object):
        x = Age(8)

        def __init__(self):
            self.x = Age(8)

    cls = MyClass()
    print(cls.__class__.__dict__['x'])
    print(cls.__dict__['x'])


def demo_5():
    class A:
        def __init__(self):
            self.foo = 'abc'

        def foo(self):
            return 'xyz'

    print(A().foo)
    print(dir(A.foo))
    # print(A.foo)
    # print(A.foo.__get__(A)())


def demo_6():
    class Age(object):

        def __init__(self, age):
            self.val = age

        def __getattribute__(self, item):
            print('Age.__getattribute__')
            return super(Age, self).__getattribute__(item)

        def __get__(self, obj, objtype):
            print('Age.__get__')
            return self.val

        def __set__(self, instance, value):
            pass

    class MyClass(object):
        x = Age(10)

        def __init__(self):
            self.x = 5

    cls = MyClass()

    print(cls.x)
    print(cls.__dict__)

def demo_7():

    class A(object):

        def foo(self):
            return 'xyz'

    def fpp():
        return 'zzz'

    # 三者等价
    print(A.foo)
    print(A.__dict__['foo'])
    # object=None,所以原来叫做没有和类绑定的方法 Unbound method
    # 现在全部称为 function
    print(A.__dict__['foo'].__get__(None, A))

    print(A().foo)
    print(A.foo.__get__(A))
    print(A.foo.__get__(A)())  # 类似于 A.foo(A)
    print(fpp.__get__(A))
    print(fpp.__get__(A)())  # 类似于fpp(A)


if __name__ == '__main__':
    demo_7()
