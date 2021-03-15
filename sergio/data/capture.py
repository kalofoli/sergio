
import numpy as np
import pandas as pd

from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.widgets as mplwidgets
import itertools
import functools
from types import SimpleNamespace
from matplotlib.backend_bases import MouseButton
from matplotlib import pyplot as plt
from typing import NamedTuple
import typing
from sergio.data.bundles.entities import _resolve_dataframe_index

__version__ = '1.0.0'

def _make_marker(idx, num, r=1, steps=10):
    r = 1
    t_beg, t_end = (np.r_[0,1]+idx)/num
    t = np.linspace(t_beg*2*np.pi,t_end*2*np.pi,steps)+np.pi/2
    x = np.r_[0, np.cos(t)*r]
    y = np.r_[0, np.sin(t)*r]
    return np.c_[x,y]

class OutputContext():
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass


def capture_output(fn):
    functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._output_context:
            fn(self, *args, **kwargs)
    return wrapper

class Point(NamedTuple):
    x: float
    y: float
    selected: bool = False

class LookupResult:
    def __init__(self, data, index):
        self._data = data
        self._index = index
    @property
    def index(self): return self._index
    @property
    def name(self): return self._data._df_attrs.columns[self._index]
    @property
    def data(self): return self._data._df_attrs.iloc[:,self._index]
    @property
    def selected(self): return self._data._df_cols.selected[self._index]
    @selected.setter
    def selected(self, value): self._data._set_column_param(self.index, 'selected', value)
    @property
    def dtype(self): return self._data._df_cols.dtype[self._index]
    @dtype.setter
    def dtype(self, value): self._data._set_column_param(self.index, 'dtype', value)
    @property
    def default(self): return self._data._df_cols.default[self._index]
    @default.setter
    def default(self, value): self._data._set_column_param(self.index, 'default', value)
    @property
    def visible(self): return self._data._df_cols.visible[self._index]
    @visible.setter
    def visible(self, value): self._data._set_column_param(self.index, 'visible', value)

def _make_range(v, v_min, v_max, v_other=0):
    if v.dtype.kind == 'i':
        M = v.max()
        m = v.min()
        if m == M:
            m = M - 1
        sz = (v-m)/(M-m)*(v_max-v_min)+v_min
    elif v.dtype.kind == 'b':
        sz = np.empty(len(v), float)
        sz[ v] = v_max
        sz[~v] = v_min
    elif v.dtype.kind == 'f':
        idl_ok = ~(np.isnan(v) | np.isinf(v))
        if not np.all(idl_ok):
            v_ok = v[idl_ok]
        else:
            v_ok = v
        M = v_ok.max()
        m = v_ok.min()
        sz = (v-m)/(M-m)*(v_max-v_min)+v_min
        sz[~idl_ok] = v_other
    elif isinstance(v.dtype, pd.CategoricalDtype):
        sz = _make_range(v.cat.codes.values, v_min, v_max, v_other)
    else:
        sz = np.full(len(v), v_min)
    return sz

def _roarray(what):
    if what is None:
        return None
    a = np.array(what)
    a.setflags(write=False)
    return a

class DatasetCaptureState:
    def __init__(self, df_attrs =None, df_points=None, df_cols=None):
        self._df_attrs = df_attrs if df_attrs is not None else pd.DataFrame()
        self._df_points = df_points
        if df_points is None:
            df_points = [Point(0,0)]
        self._df_points = pd.DataFrame(df_points)
        self._df_cols = df_cols if df_cols is not None else \
            pd.DataFrame({'selected':[],'visible':[],'default':[],'dtype':[]}).astype({'selected':bool,'visible':bool})
        self._capture = None
        self._last = SimpleNamespace(dtype='bool',value='0', name='name', file='data.csv')

    def _set_capture(self, capture):
        self._capture = capture
    def lookup_attribute(self, what) -> LookupResult:
        """Return a set of information about an attribute by index or name"""
        try:
            idx = _resolve_dataframe_index(self._df_attrs, what)
            return LookupResult(self, idx)
        except KeyError:
            return None
    @property
    def last(self): return self._last
    def point_add(self, point):
        idx = max(0, self._df_attrs.index.max() + 1)
        dtypes = dict(self._df_cols.dtype.items())
        row = pd.DataFrame([self._df_cols.default], columns=self._df_cols.index, index=[idx])
        self._df_attrs = self._df_attrs.append(row).astype(dtypes)
        self._df_points.loc[idx] = pd.Series(point._asdict())
        self._capture.cb_draw_points()
    def point_remove_index(self, idx=None):
        if idx is None:
            idx = np.nonzero(self.points_selected)[0]
        df_idx = self._df_points.index[idx]
        self._df_attrs.drop(index=df_idx, inplace=True)
        self._df_points.drop(index=df_idx, inplace=True)
        self._capture.cb_draw_points()
    @property
    def points_selected(self): return _roarray(self._df_points['selected'])
    @points_selected.setter
    def points_selected(self, value):
        sel_orig = self.points_selected
        if not np.all(sel_orig==value):
            self._df_points['selected'] = value
            self._capture.cb_draw_points()
    def point_toggle_selected(self, idx):
        self._df_points['selected'][idx] = ~self._df_points['selected'][idx]
        self._capture.cb_draw_points()
    @property
    def point_array(self): return np.stack(self.point_coords, axis=1)
    @property
    def point_coords(self): return self._df_points['x'], self._df_points['y']

    @property
    def column_selected(self): return _roarray(self._df_cols.selected)
    @column_selected.setter
    def column_selected(self, value):
        if value != self.column_selected:
            self._df_cols.selected = value
            self._capture.cn_draw_widget()
    @property
    def column_default(self): return list(self._df_cols.default)
    @property
    def column_visible(self): return _roarray(self._df_cols.visible)
    @property
    def columns(self): return self._df_attrs.columns
    @property
    def visible_columns(self) -> typing.Tuple[str]:
        return tuple(itertools.compress(self.columns, self.column_visible))
    @property
    def selected_columns(self) -> typing.Tuple[str]:
        return tuple(itertools.compress(self.columns, self.column_selected))

    def _set_column_param(self, idx, key, value):
        idx_key = self._df_cols.columns.get_loc(key)
        self._df_cols.iloc[idx, idx_key] = value
        if key == 'visible':
            if not bool(value):
                name = self._df_attrs.columns[idx]
                self._capture.cb_hide_column(name)
        if key in {'visible','selected'}:
            self._capture.cb_draw()
    
    def new_column(self, name, dtype, value=0):
        visible = not self.column_visible.sum()
        selected = not np.any(self.column_selected)
        df_col_row = pd.DataFrame({'selected':selected, 'visible':visible, 'default':value, 'dtype':dtype}, index=[name])
        self._df_cols = self._df_cols.append(df_col_row)
        self._df_attrs[name] = pd.Series([value]*self._df_attrs.shape[0], dtype=dtype)
        self._capture.cb_draw()
    def set_values(self, value):
        for name, row in self._df_cols[self._df_cols.selected].iterrows():
            series = self._df_attrs.loc[:,name]
            if isinstance(series.dtype, pd.CategoricalDtype):
                c = series.values
                if value not in c.categories:
                    c.add_categories([value], inplace = True)
                series.loc[self.points_selected] = value
                self._df_attrs.loc[:, name] = series
            else:
                va = pd.Series(value, dtype=series.dtype).astype(series.dtype).values[0]
                self._df_attrs.loc[self.points_selected, name] = va
        self._capture.cb_draw()
    def del_column(self, what=None):
        if what is not None:
            alu = self.lookup_attribute(what)
            
            names = [alu.name] if alu is not None else []
        else:
            names = self.selected_columns
        if names:
            for name in names:
                self._capture._remove_artists(name)
                self._df_attrs.pop(name)
                self._df_cols.drop(name, inplace=True, axis=0)
            self._capture.cb_draw()
  
    def get_point_attrs(self, index):
        return self._df_attrs.iloc[index,:]
    
    def point_index2ident(self, index):
        return self._df_points.index[index]

    def save(self, file):
        import datetime
        with pd.HDFStore(file, 'w') as hs:
            hs.put('attributes', self._df_attrs.reset_index(drop=True), format='table')
            hs['target'] = self._df_points[['x','y']].reset_index(drop=True)
            hs['info'] = pd.Series({'version':__version__, 'time':datetime.datetime.now()})
            hs['/meta/capture/points'] = self._df_points
            hs.put('/meta/capture/attributes', self._df_attrs, 't')
            hs['/meta/capture/columns'] = self._df_cols
            last = self._last
            last.limits = self._capture.limits
            df_last = pd.DataFrame(dict(zip(['key','value'], zip(*last.__dict__.items()))))
            hs['/meta/capture/last'] = df_last
            
            
    
    def load(self, file):
        with pd.HDFStore(file, 'r') as hs:
            self._df_attrs = hs['/meta/capture/attributes']
            self._df_points = hs['meta/capture/points']
            self._df_cols = hs['/meta/capture/columns']
            df_last = hs['/meta/capture/last']
            self._last = SimpleNamespace(**dict(zip(df_last.key, df_last.value)))
        self._capture.cb_reset(*self._last.limits)
        self._capture.cb_draw()
    
class DatasetCapture:
    def __init__(self, state=None, fig=None, output_context=OutputContext()):
        self._state = state if state is not None else DatasetCaptureState()
        self._state._set_capture(self)
        
        self._output_context = output_context
        self._controls = SimpleNamespace()
        
        self._grid = plt.GridSpec(1,2, wspace=0.4, hspace=0.3, width_ratios=[4,1])
        self._grid_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self._grid[0], hspace=0.1, height_ratios=[9,1])
        self._fig = fig if fig is not None else plt.figure()
        self._alpha_nosel = 0.4
        self._size = 30
        self._size_min = 5
        self._size_nan = 1
        self._mods = SimpleNamespace(shift=False, ctrl=False, sup=False)
        self.cid = None
        self._ax_plt = None
        self._collection = None
        self._collections = {}
        self._annotation = None
        
        self.cb_draw()
        self.register()
        

    @capture_output
    def on_column_selected(self, w, label):
        self._state.lookup_attribute(label).selected ^= True
    @capture_output
    def on_visible(self, w, label):
        alu = self._state.lookup_attribute(label)
        alu.visible = ~alu.visible
    @capture_output
    def on_add(self, name, dtype, value):
        self._state.new_column(name, dtype, value)
    @capture_output
    def on_set(self, value):
        self._state.set_values(value)
    @capture_output
    def on_rem(self, name):
        self._state.del_column(None)

    @capture_output
    def on_size(self, w, size):
        self._size = size
        self.cb_draw_points()
        
    @capture_output
    def on_key_press(self, event):
        if event.key == 'shift':
            self._mods.shift = True
        elif event.key == 'control':
            self._mods.ctrl = True
        elif event.key == 'super':
            self._mods.sup = True
        elif event.key == 'delete':
            self._state.point_remove_index()

    @capture_output
    def on_key_release(self, event):
        if event.key == 'shift':
            self._mods.shift = False
        elif event.key == 'control':
            self._mods.ctrl = False
        elif event.key == 'super':
            self._mods.sup = False
        elif event.key == 'escape':
            self._state.points_selected = False
            self.cb_draw_points()
    @capture_output
    def on_pick(self, event):
        mevent = event.mouseevent
        if mevent.inaxes is not None and mevent.inaxes.get_label() == 'plot':
            idx = event.ind[0]
            if mevent.button==MouseButton.RIGHT:
                self._state.point_remove_index(idx)
            elif mevent.button == MouseButton.LEFT and self._mods.shift:
                self._state.point_toggle_selected(idx)
    @capture_output
    def on_click(self, event):
        if event.inaxes is not None and event.inaxes.get_label() == 'plot' and not self._mods.shift:
            if event.button == MouseButton.LEFT:
                self._state.point_add(Point(event.xdata, event.ydata))
    @capture_output
    def on_points_selected(self, verts):
        path = mpl.path.Path(verts)
        offsets = self._state.point_array
        sels = np.array(self._state.points_selected)
        idl_contained = path.contains_points(offsets)
        if self._mods.shift:
            sels |= idl_contained
        elif self._mods.ctrl:
            sels &= ~idl_contained
        elif self._mods.sup:
            sels ^= idl_contained
        else:
            sels = idl_contained
        self._state.points_selected = sels
                
    @capture_output
    def on_hover(self, event):
        if event.inaxes is not None and event.inaxes.get_label() == 'plot':
            visible = self._annotation.get_visible()
            is_within = False
            for c in itertools.chain((self._collection,), self._collections.values()):
                is_within, info = c.contains(event)
                if is_within:
                    break
            if is_within:
                idx = info['ind'][0]
                self.cb_draw_annot(idx)
            else:
                if visible:
                    self.cb_draw_annot()

    def mark_values(self, idx, num, name):
        c = self._collections.get(name)
        v = self._state.lookup_attribute(name).data
        if c is None:
            c = self._collections[name] = self._ax_plt.scatter([],[], picker=True)
        xy = self._state.point_array
        verts = _make_marker(idx, num)
        c.set_offsets(xy)
        sz = _make_range(v, self._size_min, self._size, self._size_nan)
        c.set_sizes(sz)
        p = mpl.path.Path(verts)
        c.set_paths([p])
        c.set_visible(True)
    
    

    def cb_draw_widget(self):
        columns = self._state.columns
        gs = self._grid[1]
        controls = self._controls
        grid = gridspec.GridSpecFromSubplotSpec(9, 3, subplot_spec=gs, height_ratios=[7,7,1,1,1,1,1,1,1], hspace=0.1)
        controls.w_vis = mplwidgets.CheckButtons(plt.subplot(grid[0,:]), labels=columns, actives=self._state.column_visible)
        controls.w_sel = mplwidgets.CheckButtons(plt.subplot(grid[1,:]), labels=columns, actives=self._state.column_selected)
        controls.w_type = mplwidgets.TextBox(plt.subplot(grid[2,:]), label='dtype', initial=self._state.last.dtype)
        controls.w_name = mplwidgets.TextBox(plt.subplot(grid[3,:]), label='name', initial = self._state.last.name)
        controls.w_val = mplwidgets.TextBox(plt.subplot(grid[4,:]), label='val', initial = self._state.last.value)
        controls.w_add = mplwidgets.Button(plt.subplot(grid[5,0]), label='+')
        controls.w_set = mplwidgets.Button(plt.subplot(grid[5,1]), label='=')
        controls.w_rem = mplwidgets.Button(plt.subplot(grid[5,2]), label='-')
        controls.w_sz = mplwidgets.Slider(plt.subplot(grid[6,:]), valmin=1, valmax=500, valinit=self._size, label='size')
        controls.w_file = mplwidgets.TextBox(plt.subplot(grid[7,:]), label='file', initial = self._state.last.file)
        controls.w_load = mplwidgets.Button(plt.subplot(grid[8,:2]), label='LD')
        controls.w_save = mplwidgets.Button(plt.subplot(grid[8,2]), label='SV')
        
        controls.w_sel.on_clicked(lambda x:self.on_column_selected(controls.w_sel, x))
        controls.w_vis.on_clicked(lambda x:self.on_visible(controls.w_vis, x))
        controls.w_add.on_clicked(lambda x:self.on_add(controls.w_name.text, controls.w_type.text, controls.w_val.text))
        controls.w_set.on_clicked(lambda x:self.on_set(controls.w_val.text))
        controls.w_rem.on_clicked(lambda x:self.on_rem(controls.w_name.text))
        controls.w_sz.on_changed(lambda x:self.on_size(controls.w_sz, x))
        controls.w_save.on_clicked(lambda x:self._state.save(controls.w_file.text))
        controls.w_load.on_clicked(lambda x:self._state.load(controls.w_file.text))
        
        cols = self._state.columns
        if len(cols):
            def make_def_setter(index):
                def setter(value):
                    self._state.lookup_attribute(index).default = value
                return setter 
            gs = self._grid_left[1]
            grid = gridspec.GridSpecFromSubplotSpec(1, len(cols), subplot_spec=gs, hspace=0.1)
            controls.w_defs = []
            for col in cols:
                alu = self._state.lookup_attribute(col)
                w_def = mplwidgets.TextBox(plt.subplot(grid[alu.index]), label=alu.name, initial = alu.default)
                w_def.on_submit(make_def_setter(alu.index))
                controls.w_defs.append(w_def)

        def make_par_setter(name):
            @capture_output
            def setter(value):
                setattr(self._state.last, name, value)
            return setter
        controls.w_file.on_submit(make_par_setter('file'))
        controls.w_type.on_submit(make_par_setter('dtype'))
        controls.w_name.on_submit(make_par_setter('name'))
        controls.w_val.on_submit(make_par_setter('value'))
        self._fig.canvas.draw_idle()

    def _highlight_collection(self, c):
        idl = self._state.points_selected
        fc = c.get_facecolors()[0,:]
        n = c.get_offsets().shape[0]
        if np.any(idl):
            fc = np.tile(fc, (n, 1))
            fc[~idl, -1] = self._alpha_nosel
            fc[ idl, -1] = 1
        else:
            fc[-1] = 1
        c.set_facecolors(fc)
    
    def cb_draw_points(self):
        if self._ax_plt is None:
            self._ax_plt = plt.subplot(self._grid_left[0], label='plot', autoscale_on=True, xlim=[0,1], ylim=[0,1])
        if self._collection is None:
            self._collection = self._ax_plt.scatter([],[], picker=True)
            self._controls.lasso = mplwidgets.LassoSelector(ax = self._ax_plt, onselect=self.on_points_selected, button=MouseButton.RIGHT)
            self._annotation = self._ax_plt.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            self._annotation.set_visible(False)
        x, y = self._state.point_coords
        idl_visible = self._state.column_visible
        num_vis = idl_visible.sum()
        if num_vis == 0:
            c = self._collection
            c.set_offsets(np.c_[x,y])
            c.set_sizes([self._size])
            c.set_visible(True)
            self._highlight_collection(c)
        else:
            self.cb_hide_column()
            for idx, name in enumerate(self._state.visible_columns):
                self.mark_values(idx, num_vis, name)
                c = self._collections[name]
                self._highlight_collection(c)
        self._fig.canvas.draw_idle()

    def cb_hide_column(self,name=None):
        if name is None:
            c = self._collection
        else:
            c = self._collections[name]
        c.set_visible(False)
    def _remove_artists(self, name):
        self._collections[name].remove()
        del self._collections[name]
        
    def cb_draw(self):
        self.cb_draw_points()
        self.cb_draw_widget()
        self._fig.canvas.draw_idle()
    
    def register(self):
        if self.cid is None:
            cid = self.cid = SimpleNamespace()
            canvas = self._fig.canvas
            cid.pick = canvas.mpl_connect('pick_event', self.on_pick)
            cid.click = canvas.mpl_connect('button_press_event', self.on_click)
            cid.key_press = canvas.mpl_connect('key_press_event', self.on_key_press)
            cid.key_release = canvas.mpl_connect('key_release_event', self.on_key_release)
            cid.hover = canvas.mpl_connect("motion_notify_event", self.on_hover)
    
    def cb_draw_annot(self, index = None):
        annot = self._annotation
        if index is not None:
            xy = self._state.point_array[index, :].flatten()
            annot.xy = xy
            data = self._state.get_point_attrs([index])
            sattr = ', '.join(f'{key}({data.iloc[:,idx].dtype})={data.iloc[0,idx]}' for idx,key in enumerate(data.columns))
            spos = f'{xy[0]:.3g},{xy[1]:.3g}'
            ident = self._state.point_index2ident(index)
            text = f'{ident}@{index}({spos}):  {sattr}'
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)
            annot.set_visible(True)
            self._ax_plt.set_title(text)
        else:
            annot.set_visible(False)
            self._ax_plt.set_title('')
    @property
    def limits(self): return self._ax_plt.get_xlim(), self._ax_plt.get_ylim()
    
    def cb_reset(self, xlim=[0,1], ylim=[0,1]):
        self._ax_plt.clear()
        self._ax_plt.set_xlim(xlim)
        self._ax_plt.set_ylim(xlim)
        self._collection = None
        self._collections = {}
        self._annotation = None
        
    @capture_output
    def message(self, *args):
        print(*args)
    def run(self):
        plt.show(block=True)
        


if __name__ == '__main__':
    df = pd.DataFrame()
    state = DatasetCaptureState()
    capture = DatasetCapture(state, fig=plt.figure())
    plt.show(block=True)
    
    
    