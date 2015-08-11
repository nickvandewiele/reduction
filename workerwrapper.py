class WorkerWrapper(object): 
     __name__ = 'WorkerWrapper' 

     def __init__(self, myfn): 
         self.myfn = myfn 

     def __call__(self, *args, **kwargs): 
         try: 
             return self.myfn(*args, **kwargs) 
         except: 
             type, value, tb = sys.exc_info() 
             lines = traceback.format_exception(type, value, tb) 
             print ''.join(lines) 
             raise 