

def _resolve_dataframe_index(df, what):
    if isinstance(what, (np.int, int)):
        return what
    else:
        return df.columns.get_loc(what)

        
from sergio.data.capture import DatasetCapture

if __name__ == '__main__':
    capture = DatasetCapture()
    capture.run()
    
    
    