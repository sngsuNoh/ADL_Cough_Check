import librosa
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog as fd, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sound import load_audio
from sound.level import calculate_rms, rms2db, calculate_lamax, calculate_laeq, find_peaks, find_spike_indices, lamax_adjustment
from sound.weighting import A_weighting

audio_info = {
    'generator': iter([]),
    'wav': np.array([]),
    'sr': 0,
    'spl': np.array([]),
    'len': 1.0,
    'frame_start_t': 0.0,
    'processed_t': 0.0,
}

analysis_info = {
    'beg': 0.0,
    'end': 0.0,
    'cough': 0,
    'prev_laeq':[],
    'cough_size' : 15
}

root = tk.Tk()
root.title("Cough Check")
root.resizable(False, False)

frame = tk.Frame(root)
frame.pack()

figure = plt.figure(figsize=(13, 5), dpi=90)
spl_ax = figure.add_subplot(111)
spl_ax.set_title('Sound Pressure Level (dBA)')
spl_ax.set_ylabel('dBA')
spl_ax.set_xlabel('time (s)')
spl_ax.yaxis.grid()

plot_canvas = FigureCanvasTkAgg(figure, frame)
plot_canvas.get_tk_widget().grid(column=0, row=0, columnspan=6, padx=10, pady=5)

tk.Label(frame, text='set cough size (dba)', font=('Arial', 15), fg='black').grid(column=1, row=1)
cough_size_entry = tk.Entry(frame)
cough_size_entry.grid(column=1, row=2)

tk.Label(frame, text='check range (beg)', font=('Arial', 15), fg='black').grid(column=2, row=1)
tk.Label(frame, text='check range (end)', font=('Arial', 15), fg='black').grid(column=2, row=2)
cough_check_beg_str = StringVar()
cough_check_end_str = StringVar()
cough_check_beg_entry = tk.Entry(frame, textvariable=cough_check_beg_str)
cough_check_beg_entry.grid(column=3, row=1)
cough_check_end_entry = tk.Entry(frame, textvariable=cough_check_end_str)
cough_check_end_entry.grid(column=3, row=2)

laeq_str = StringVar()
laeq_str.set('LAeq (dBA): 0')
tk.Label(frame, textvariable=laeq_str, font=('Arial', 15), fg='black').grid(column=5, row=1, pady=5)

lamax_str = StringVar()
lamax_str.set('LAmax (dBA): 0')
tk.Label(frame, textvariable=lamax_str, font=('Arial', 15), fg='black').grid(column=5, row=2, pady=5)

cough_str = StringVar()
cough_str.set('Counted Cough : 0')
tk.Label(frame, textvariable=cough_str, font=('Arial', 15), fg='black').grid(column=4, row=2, pady=5)

def open_audio_file():
    filetypes = (
        ('raw audio file', '.wav'),
    )
    filename = fd.askopenfilename(filetypes=filetypes)
    if filename != '':
        # frame length adjustment
        stream, sr = load_audio(filename, block_len=20)
        audio_info['generator'] = stream
        audio_info['sr'] = sr
        audio_info['processed_t'] = 0.0
        audio_info['frame_start_t'] = 0.0
        analysis_info['cough'] = 0
        analysis_info['prev_laeq'] = []
        if cough_size_entry.get():
            analysis_info['cough_size'] = int(cough_size_entry.get())

        next_audio_frame()
        
def next_audio_frame():
    wav = next(audio_info['generator'], None)
    if wav is None:
        return
    wav, sr = A_weighting(wav, audio_info['sr'])
    rms = calculate_rms(wav, sr)
    spl = rms2db(rms)

    audio_info['wav'] = wav
    audio_info['spl'] = spl[0]
    audio_info['len'] = librosa.get_duration(y=wav, sr=sr)
    audio_info['frame_start_t'] = audio_info['processed_t']
    audio_info['processed_t'] += audio_info['len']
    
    cough_check_beg_str.set(f"{audio_info['frame_start_t']:.7f}")
    cough_check_end_str.set(f"{audio_info['processed_t']:.7f}")
    

    update_laeq_lamax_label(redraw=False)
    
    draw_spl()
    
def draw_spl():
    draw_spl_data(audio_info['spl'], audio_info['sr'])
    
def draw_spl_data(spl, sr:int):
    start_t = audio_info['frame_start_t']
    times = get_times(spl, sr, start_t)

    spl_ax.cla()
    spl_ax.plot(times, lamax_adjustment(spl))

    spl_ax.set_xlim(xmin=times[0])
    spl_ax.set_xlim(xmax=times[-1])
    spl_ax.set_xticks(get_ticks(times, interval=get_interval(times[-1])))

    spl_ax.set_title('Sound Pressure Level (dBA)')
    spl_ax.set_ylabel('dBA')
    spl_ax.set_xlabel('time (s)')
    spl_ax.yaxis.grid()

    plot_canvas.draw()

def get_interval(end_t: float) -> float:
    if math.log10(n) >= 2: return 0.5*(int(log10(n)+1))
    else: return 1.0

def get_ticks(times, interval: float = 1.0):
    start_t = audio_info['frame_start_t']
    end_t = audio_info['processed_t']
    start_t = math.ceil(start_t) + interval if math.ceil(start_t) - start_t < interval else math.ceil(start_t)
    end_t = math.floor(end_t) + interval if end_t - math.floor(end_t) > interval else round(end_t - interval)
    ticks = np.arange(start_t, end_t, interval)

    if ticks.size == 0:
        start = []
        end = []
    else:
        start = [] if round(times[0], 1) == round(ticks[0], 1) else [round(times[0], 1)]
        end = [] if round(times[-1], 1) == round(ticks[-1], 1) else [round(times[-1], 1)]

    ticks = np.concatenate((start, ticks, end))
    return ticks

def get_times(spl, sr: int, start: float):
    times = librosa.times_like(spl, sr=sr)
    return times + start

def update_laeq_lamax_label(redraw: bool = True):
    spl_len = audio_info['spl'].shape[0] 
    if spl_len == 0:
        return

    start_t = audio_info['frame_start_t']

    try:
        beg_sec = float(cough_check_beg_str.get())
    except ValueError:
        beg_sec = start_t
    try:
        end_sec = float(cough_check_end_str.get())
    except ValueError:
        end_sec = audio_info['len']
        cough_check_end_str.set(f"{audio_info['len']}")

    beg_sec = np.clip(beg_sec, start_t, audio_info['processed_t'])
    end_sec = np.clip(end_sec, start_t, audio_info['processed_t']) 

    beg = round(spl_len * ((beg_sec - start_t) / audio_info['len'])) 
    end = round(spl_len * ((end_sec - start_t) / audio_info['len'])) 
    if beg > end:
        beg, end = end, beg
        beg_sec, end_sec = end_sec, beg_sec
    elif beg == end:
        end += 1

    beg_sec = ((beg / spl_len) * audio_info['len'] + start_t)
    end_sec = ((end / spl_len) * audio_info['len'] + start_t)

    cough_check_beg_str.set(f'{beg_sec:.7f}')
    cough_check_end_str.set(f'{end_sec:.7f}')

    laeq = calculate_laeq(audio_info['spl'], frames_beg=beg, frames_end=end)
    laeq_str.set(f'LAeq (dBA): {laeq:.5f}')

    lamax = calculate_lamax(audio_info['spl'], frames_beg=beg, frames_end=end)
    lamax_str.set(f'LAmax (dBA): {lamax:.5f}')
    
    if analysis_info['prev_laeq']:
        if lamax > analysis_info['prev_laeq'][-1] + analysis_info['cough_size']:
            analysis_info['cough'] += 1
    analysis_info['prev_laeq'].append(laeq)
    
    cough_str.set(f'Counted Cough : {analysis_info["cough"]}')

    analysis_info['beg'] = beg_sec
    analysis_info['end'] = end_sec

    if redraw:
        draw_spl()

ttk.Button(
    frame,
    text='cough check start',
    command=open_audio_file,
).grid(column=4, row=1)

ttk.Button(
    frame,
    text='open an audio file',
    command=open_audio_file
).grid(column=0, row=1, padx=5)

ttk.Button(
    frame,
    text='next',
    command=next_audio_frame
).grid(column=0, row=2, padx=5)



root.mainloop()
