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
            self.x = 5

        def __getattribute__(self, item):
            print('MyClass.__getattribute__',item)
            return super(MyClass, self).__getattribute__(item)

        def __getattr__(self, item):
            print('MyClass.__getattr__',item)
            raise AttributeError(f'{type(self).__name__} object had not attribute \'{item}\'')

        def __setattr__(self, key, value):
            print('MyClass.__setattr__',key)
            # self.key = value # 错误
            self.__dict__[key] = value

    cls = MyClass()
    print(cls.__dict__)
    print(cls.__class__.__dict__)


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




def demo_8():

    class Property(object):
        "Emulate PyProperty_Type() in Objects/descrobject.c"

        def __init__(self, fget=None, fset=None, fdel=None, doc=None):
            print('-----')
            print('fget', fget)
            print('fset', fset)
            print('fdel', fdel)
            print('-----')
            self.fget = fget
            self.fset = fset
            self.fdel = fdel
            if doc is None and fget is not None:
                doc = fget.__doc__
            self.__doc__ = doc

        def __get__(self, obj, objtype=None):
            print('Property.__get__')
            if obj is None:
                return self
            if self.fget is None:
                raise AttributeError("unreadable attribute")
            return self.fget(obj)

        def __set__(self, obj, value):
            print('Property.__set__',obj)
            if self.fset is None:
                raise AttributeError("can't set attribute")
            self.fset(obj, value)

        def __delete__(self, obj):
            print('Property.__delete__')
            if self.fdel is None:
                raise AttributeError("can't delete attribute")
            self.fdel(obj)

        def getter(self, fget):
            print('Property.getter')
            return type(self)(fget, self.fset, self.fdel, self.__doc__)

        def setter(self, fset):
            print('Property.setter')
            return type(self)(self.fget, fset, self.fdel, self.__doc__)

        def deleter(self, fdel):
            print('Property.deleter')
            return type(self)(self.fget, self.fset, fdel, self.__doc__)

    class C(object):


        def __init__(self):
            self._x = None

        @Property
        def x(self):
            print('C.property')
            return self._x

        @x.setter
        def x(self, value):
            print('C.setter')
            self._x = value

        @x.deleter
        def x(self):
            print('C.deleter')
            del self._x

    c=C()
    print(C.__dict__['x'])
    print(c.__dict__)
    c.x=8
    print(c.x)
    del c.x


    class C(object):
        def __init__(self):
            self._x = None


        def getx(self):
            print('C.property')
            return self._x


        def setx(self, value):
            print('C.setter')
            self._x = value


        def delx(self):
            print('C.deleter')
            del self._x

        x = Property(getx,setx,delx)

    c = C()
    c.x=8
    print(c.x)
    del c.x


    # 模拟显示过程
    class C(object):


        def __init__(self):
            self._x = None

        def getx(self):
            print('C.property')
            return self._x


        def setx(self, value):
            print('C.setter')
            self._x = value


        def delx(self):
            print('C.deleter')
            del self._x

    x_get=Property(C.getx)
    x_set=x_get.setter(C.setx)
    x_del=x_set.deleter(C.delx)

    x_set.__set__(C(),8)
    print(x_get.__get__(C()))
    x_del.__delete__(C())


if __name__ == '__main__':
    # demo_2()
    demo_8()

