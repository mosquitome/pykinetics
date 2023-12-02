import os
import numpy as np
import pandas as pd
import datetime as dt
import scipy.signal as sg
import scipy.interpolate as si
import matplotlib.pyplot as mplp
import matplotlib.gridspec as mplg
import seaborn as sb

#==============================================================================
#                             P Y K I N E T I C S
#==============================================================================

# Author        :   David A. Ellis <https://github.com/mosquitome/>
# Organisation  :   University College London

# Requirements  :   Python 3.8.11
#                   pandas 1.3.1
#                   numpy 1.20.3
#                   scipy 1.6.2
#                   matplotlib 3.7.1
#                   seaborn 0.12.2

#                          I N S T R U C T I O N S :
#
# 1. Fill out layout file (template file Layout.txt provided), leaving any
#    descriptor cells empty for vials you want to ignore (e.g. those containing
#    dead mosquitoes). Save as tab delimited in the DAMSystem3Data folder
#    (along with the activity monitor data files). Note: use as many
#    descriptors as you wish - just add a new row with 'descriptor-n:' in the
#    first column (where n is a number) and the descriptor name in the next,
#    then add a column with the data.
#
# 2. Change working directory to the DAMSystem3Data folder.
#
# 3. Set the variables below:

env_monitor = '95' # <- number (as string) of environmental monitor used in experiment
time_window = '5M' # <- window size in seconds, minutes or hours (S, M or H) to use, regardless of time used for recording e.g. '30M'
trim = 'smart' # <- amount to trim from start and end. Can either set manually, with amount to trim from start and end seperated by a comma (e.g. '10M','4H'), or can set to 'smart' to automatically trim to the first and last zt0. If setting manually but you do not wish to trim either start or end, enter None for that parameter (e.g. None,'2H'). If you don't want to trim, set trim=None,None
zt0 = '20:55:00' # <- time of dawn during entrainment. Can either be set manually in the format 'HH:MM:SS' e.g. '20:00:00' for 8pm dawn, or set to 'smart' to infer dawn from light-lux over first 24hrs (accurate to nearest minute); NOTE: THIS IS BASED ON THE TRIKINETICS COMPUTER TIME
period = '24H' # <- length of period during entrainment in seconds, minutes or hours (S, M or H) e.g. '24H'
epoch_ids = 1,2,3,4,5,6 # <- this could be a single epoch (e.g. 2) or multiple comma-seperated epochs (e.g. 1,2,5) that will be merged for some analysis. If None, all epochs will be used.
#analyse_sleep

#==============================================================================

#                         R E A D   I N   D A T A :

# A. If zt0=='smart', initially subset 1st 24hrs of data from env monitor, bin in 1min bins, infer zt0, then continue as normal

# B. If analyse_sleep==True, a seperate dataframe with all data binned to 5min bins (perion of sleep is defined as 5min of inactivity) is made, regardless of time_window parameter.

def convert_time(time, unit):
    '''converts a given time in the format 'timeUNIT', where 'time' is numeric
    and 'UNIT' is either S, M or H for seconds, minutes or hours (e.g. '10S'
    for 10 seconds), to a different unit (e.g. 'M')'. Both time and unit
    should be strings. Returns a float.'''
    if time[-1] == 'S':
        conversion = {'S':1.0,'M':1.0/60.0,'H':1.0/3600.0}
    elif time[-1] == 'M':
        conversion = {'S':60.0,'M':1.0,'H':1.0/60.0}
    elif time[-1] == 'H':
        conversion = {'S':3600.0,'M':60.0,'H':1.0}
    time_new = float(time[:-1]) * conversion[unit]
    return time_new

time_window = int(convert_time(time_window,'S')) # <- ensure time window is in seconds

def timeseries(start, end, step, inc=False):
    '''produce list of times in the format 'HH:MM:SS', starting at start (as 
    string i.e. 'HH:MM:SS'), stopping at (but not including*) end (as string 
    i.e. 'HH:MM:SS'), in units of step (as integer, in seconds i.e. 60).
    Alternatively to writing start time as a string, enter an integer of the 
    number of seconds prior to the end time you would like to use. Same goes
    for end, but use the number of seconds after start. *To include the end
    time in the returned list, set inc=True.'''
    if type(start)==str:
        t1 = dt.datetime.strptime(start,'%H:%M:%S')
    elif type(start)==int:
        t1 = dt.datetime.strptime(end,'%H:%M:%S') - dt.timedelta(seconds=start)
    else:
        print('E R R O R : start must be str or int')
        return None
    if type(end)==str:
        t2 = dt.datetime.strptime(end,'%H:%M:%S')
    elif type(end)==int:
        t2 = dt.datetime.strptime(start,'%H:%M:%S') + dt.timedelta(seconds=end)
    else:
        print('E R R O R : end must be str or int')
        return None
    if t1 >= t2:
        print('E R R O R : start must come before end')
        return None
    delta = dt.timedelta(seconds=step)
    times = []
    while t1 < t2:
        times.append(t1.time().isoformat())
        t1 += delta
    if inc==True:
        times.append(t2.time().isoformat())
    return times

def find_monitors(env_monitor=env_monitor):
    '''searches the current directory (of DAMSystem3Data type) for monitors
    that were active. returns a dictionary of these monitors with any
    timepoints where the monitor was inactive, removed.'''
    monitors = {}
    files = [i for i in os.listdir(os.getcwd()) if 'Monitor' in i]
    for file in files:
        temp = pd.read_table(file,nrows=5,header=None)
        if temp.iloc[0,3]!=51: # <- 51 flags inactive monitors (at least those that were on at start); this is better than using ==1 to prevent rows being dropped at start of some files but not others, leading to incompatible lengths
            if file.split('.')[0][-2:]==env_monitor:
                temp2 = pd.read_table(file,header=None)
                monitors['environmental'] = temp2.loc[temp2[3]==1] # <- only get rows where status was 'good'
            else:
                number = file.split('.')[0].split('Monitor')[1]
                temp2 = pd.read_table(file,header=None)
                monitors[number] = temp2.loc[temp2[3]==1] # <- only get rows where status was 'good'
    # add blank rows where there are gaps in time:
    times = list(monitors['environmental'].iloc[:10,2])
    steps = [(dt.datetime.strptime(times[j],'%H:%M:%S') - dt.datetime.strptime(times[i],'%H:%M:%S')).seconds for i,j in zip(range(0,len(times)-1),range(1,len(times)))]
    step = int(np.median(steps)) # <- average step size (in seconds) between ten measurements
    for monitor in monitors.keys():
        temp = monitors[monitor].copy()
        temp = temp.set_index(pd.to_datetime(temp[1]+' '+temp[2])).asfreq(str(step)+'S') # <- add blank rows where there are gaps in time.
#        temp.loc[:,1:2] = [[i.date().strftime('%#d %b %y'),i.time().isoformat()] for i in temp.index] # <- move date and time back to columns (from where they are at the moment in the index) # UNNECESSARY STEP AS DATE AND TIME COLUMNS WERE NEVER REMOVED WHEN SET AS INDEX
        temp[0] = temp.index
        monitors[monitor] = temp.reset_index(drop=True)
    return monitors

monitors = find_monitors()

def trim_timecourse(trim_parameter, monitor_dictionary):
    monitor_dictionary = monitor_dictionary.copy()
    keys = [i for i in monitor_dictionary.keys() if i!='environmental']
    # manual trimming...
    if type(trim_parameter)==tuple:
        trim_start = trim_parameter[0]
        trim_end = trim_parameter[1]
        # trim according to trim_start and trim_end:
        if trim_start == None and trim_end == None:
            pass
        else:
            dt_list = monitor_dictionary['environmental'].iloc[:,1:3].apply(lambda x: dt.datetime.strptime(' '.join(x),'%Y-%m-%d %H:%M:%S'), axis=1) # <- make a list of date times (used later for finding the nearest available datetime)
        if trim_end != None:
            for monitor in keys:
                end_datetime = ' '.join(monitor_dictionary[monitor].iloc[-1:,1:3].values[0]) # <- date and time at final row of data
                end_dt = dt.datetime.strptime(end_datetime,'%Y-%m-%d %H:%M:%S')
                if trim_end[-1]=='H':
                    trim_dt = dt.timedelta(hours=int(trim_end[:-1]), minutes=0, seconds=0)
                elif trim_end[-1]=='M':
                    trim_dt = dt.timedelta(hours=0, minutes=int(trim_end[:-1]), seconds=0)
                elif trim_end[-1]=='S':
                    trim_dt = dt.timedelta(hours=0, minutes=0, seconds=int(trim_end[:-1]))
                else:
                    print('E R R O R : problem with trim_end format...')
                new_dt = end_dt - trim_dt
                new_dt2 = min(dt_list, key=lambda x: abs(x - new_dt)) # <- get nearest datetime to new_dt available in dt_list (made above)
                new_idx = monitor_dictionary[monitor].loc[(monitor_dictionary[monitor][1]==new_dt2.date().isoformat()) & \
                                                          (monitor_dictionary[monitor][2]==new_dt2.time().isoformat())].index.values[0] # <- get the index of the row that is trim_end away from the end
                monitor_dictionary[monitor] = monitor_dictionary[monitor].iloc[:new_idx+1,:]
        if trim_start != None:
            for monitor in keys:
                start_datetime = ' '.join(monitor_dictionary[monitor].iloc[0:,1:3].values[0]) # <- date and time at final row of data
                start_dt = dt.datetime.strptime(start_datetime,'%Y-%m-%d %H:%M:%S')
                if trim_start[-1]=='H':
                    trim_dt = dt.timedelta(hours=int(trim_start[:-1]), minutes=0, seconds=0)
                elif trim_start[-1]=='M':
                    trim_dt = dt.timedelta(hours=0, minutes=int(trim_start[:-1]), seconds=0)
                elif trim_start[-1]=='S':
                    trim_dt = dt.timedelta(hours=0, minutes=0, seconds=int(trim_start[:-1]))
                else:
                    print('E R R O R : problem with trim_end format...')
                new_dt = start_dt + trim_dt
                new_dt2 = min(dt_list, key=lambda x: abs(x - new_dt)) # <- get nearest datetime to new_dt available in dt_list (made above)
                new_idx = monitor_dictionary[monitor].loc[(monitor_dictionary[monitor][1]==new_dt2.date().isoformat()) & \
                                                          (monitor_dictionary[monitor][2]==new_dt2.time().isoformat())].index.values[0] # <- get the index of the row that is trim_start away from the start
                monitor_dictionary[monitor] = monitor_dictionary[monitor].iloc[new_idx:,:]
    # smart trimming...
    elif trim_parameter=='smart':
        if period!='24H':
            print('E R R O R : SmartTrimming not supported for periods other than "24H"')
            return None
        if zt0 not in monitor_dictionary['environmental'][2].tolist():
            print('E R R O R : zt0 not recorded by monitors...')
            return None
        else: # <- check if idx of first and last occurences of zt0 are the same for each monitor:
            start_idx = {}
            end_idx = {}
            for monitor in monitor_dictionary.keys():
                start_idx[monitor] = monitor_dictionary[monitor].loc[monitor_dictionary[monitor][2]==zt0].iloc[0,0] # <- get the first occurrence of zt0
                end_idx[monitor] = monitor_dictionary[monitor].loc[monitor_dictionary[monitor][2]==zt0].iloc[-1,0] # <- get the last occurrence of zt0
            if all(i==start_idx['environmental'] for i in start_idx.values()) and all(i==end_idx['environmental'] for i in end_idx.values()):
                for monitor in monitor_dictionary.keys():
                    temp = monitor_dictionary[monitor].copy()
                    monitor_dictionary[monitor] = temp.loc[(temp[0]>=start_idx[monitor]) & (temp[0]<=end_idx[monitor])].reset_index(drop=True) # <- use timestamp info from start_idx and end_idx to get all rows in between start and end
                trim_start, trim_end = None, None
            else:
                print('E R R O R : Cannot use SmartTrim...')
                return None
    # error in trim parameter...
    else:
        print('E R R O R : problem with trim parameter...')
        return None
    return monitor_dictionary

monitors = trim_timecourse(trim, monitors)

def organise_timecourse(monitor_dictionary):
    ''' organise the timecourse information in a dictionary of monitors 
    produced from find_monitors. This includes creating empty rows for missing
    time points and changing the window size according to time_window. Changes
    are made to the specified dictionary in-place.'''
    times = list(monitor_dictionary['environmental'].iloc[:10,2])
    steps = [(dt.datetime.strptime(times[j],'%H:%M:%S') - dt.datetime.strptime(times[i],'%H:%M:%S')).seconds for i,j in zip(range(0,len(times)-1),range(1,len(times)))]
    step = int(np.median(steps)) # <- average step size (in seconds) between ten measurements
    # adjust window size according to time_window:
    keys = [i for i in monitor_dictionary.keys() if i!='environmental']
    if step == time_window:
        pass
    elif step % time_window == 0: # <- is current step size larger than required time window, but divisible by it.
        nbins = int(float(step)/float(time_window)) # <- from the time length of each bin (step), work out the number of bins to split into
        for monitor in keys:
            # expand (more bins, smaller timeframes):
            temp = monitor_dictionary[monitor].copy()
            temp[list(range(10,42))] = temp[list(range(10,42))].stack().apply(lambda x: [float(x)/float(nbins) for y in range(nbins)]).unstack() # <- for all columns with activity data in (columns 10 to 41), take the current value and split it evenly amongtst nbins.
            temp[2] = temp[2].apply(lambda x: timeseries(step,x,time_window,inc=True)[1:]) # <- update time column
            monitor_dictionary[monitor] = temp.explode([2] + list(range(10,42))).infer_objects().reset_index(drop=True)
    elif time_window % step == 0: # <- is current step size smaller than required time window, but a factor of it.
        nbins = int(float(time_window)/float(step)) # <- from the required time length of each bin (time_window) and the current size (step), work out the number of bins to merge from
        for monitor in keys:
            # collapse (fewer bins, larger timeframes):
            temp = monitor_dictionary[monitor].copy()
            a = temp.iloc[::nbins,0:10].reset_index(drop=True)
            b = temp.iloc[:,10:42].groupby(np.arange(len(temp))//nbins).sum(min_count=1).astype(float) # <- min_count prevents NaN being treated as zero
            monitor_dictionary[monitor] = pd.concat([a,b],axis=1).dropna(subset=[1]).infer_objects()
    elif time_window < step or step < time_window: # <- cases where step size is not divisible by time window or vice versa.
        tbins = int(np.gcd(step,time_window)) # <- first, scale the data down into bins of the smallest denominator
        nbins = int(float(step)/float(tbins)) # <- from the time length of each bin (tbins), work out the number of bins to split into
        for monitor in keys:
            # expand (more bins, smaller timeframes):
            temp = monitor_dictionary[monitor].copy()
            temp[list(range(10,42))] = temp[list(range(10,42))].stack().apply(lambda x: [float(x)/float(nbins) for y in range(nbins)]).unstack() # <- for all columns with activity data in (columns 10 to 41), take the current value and split it evenly amongtst nbins.
            temp[2] = temp[2].apply(lambda x: timeseries(step,x,tbins,inc=True)[1:]) # <- update time column
            temp = temp.explode([2] + list(range(10,42))).infer_objects().reset_index(drop=True)
            # collapse (fewer bins, larger timeframes):
            nbins2 = int(float(time_window)/float(tbins)) # <- the number of rows in the new scaled down dataset to collapse back into one row
            a = temp.iloc[::nbins2,0:10].reset_index(drop=True)
            b = temp.iloc[:,10:42].groupby(np.arange(len(temp))//nbins2).sum(min_count=1) # <- min_count prevents NaN being treated as zero
            monitor_dictionary[monitor] = pd.concat([a,b],axis=1).dropna(subset=[1]) # <- dropna removes any final, incomplete bin
    else:
        print('E R R O R : Problem with time windows...')
        return None
    # ensure datatypes are uniform, regardless of processing:
    for monitor in keys:
        monitor_dictionary[monitor] = monitor_dictionary[monitor].astype({i: 'float64' for i in range(3,42)})

organise_timecourse(monitors) # <- in-place function

def find_layouts(monitor_dictionary):
    '''Gets layout file from working directory (Layout.txt). Save layout file
    as tab delimited. For empty vials, or vials where you want to exclude data
    (e.g. those containing dead mosquitoes), leave all description rows for
    that file blank, or enter NA for each field.'''
    if 'Layout.txt' not in os.listdir(os.getcwd()):
        print('E R R O R : Layout file missing...')
        return None
    layouts = {}
    descriptors = {}
    with open('Layout.txt') as file:
        count = 0
        for line in file:
            row_start = line.rstrip().split('\t')[0]
            if 'descriptor' in row_start:
                count += 1
                name = line.rstrip().split(':')[0]
                value = line.rstrip().split('\t')[1]
                descriptors[name] = value
            else:
                break
        temp = pd.read_table('Layout.txt',skiprows=range(count),usecols=['monitor','vial'] + [i for i in descriptors.keys()],dtype={'monitor':str})
    for monitor in monitor_dictionary.keys():
        if monitor!='environmental':
            layouts[monitor] = temp.loc[temp['monitor']==monitor]
    return layouts, descriptors
            
layouts, descriptors = find_layouts(monitors)

def interpret_layouts(monitor_dictionary, layout_dictionary, descriptor_dictionary):
    '''interprets layout files and maps information to data from monitors'''
    temp = {}
    data = pd.DataFrame()
    lookup = {i: j for i,j in zip(range(1,33),range(10,42))}
    for monitor in layout_dictionary.keys():
        temp[monitor] = monitor_dictionary[monitor].copy()
        temp[monitor].rename(columns={0:'index',1:'date',2:'time',9:'light-bin'},inplace=True)
        temp[monitor]['light-lux'] = temp[monitor]['index'].map(monitor_dictionary['environmental'].set_index(0)[11])
        temp[monitor]['temperature'] = temp[monitor]['index'].map(monitor_dictionary['environmental'].set_index(0)[16])
        temp[monitor]['humidity'] = temp[monitor]['index'].map(monitor_dictionary['environmental'].set_index(0)[21])
        
        for vial in range(1,33):
            temp2 = layout_dictionary[monitor].loc[layout_dictionary[monitor]['vial']==vial]
            if temp2.isnull().values.any()==True:
                temp[monitor] = temp[monitor].drop(columns=[lookup[vial]]) # <- drop data for any vials without layout info (i.e. those with dead mosquitoes or empty vials)
            else:
                name = ''
                for idx,descriptor in enumerate(sorted(descriptor_dictionary)):
                    if idx > 0:
                        name += '_'
                    name += temp2[descriptor].values[0]
                name += '_' + str(vial)
                temp[monitor].rename(columns={lookup[vial]:name},inplace=True)
        
        columns = [i for i in temp[monitor].columns if isinstance(i, str)]
        ids = ['index','date','time','light-bin','light-lux','temperature','humidity']
        values = [i for i in columns if i not in ids]
        melted = temp[monitor].melt(value_vars=values,id_vars=ids)
        melted['monitor'] = monitor
        data = data.append(melted).reset_index(drop=True)
    for idx,descriptor in enumerate(sorted(descriptor_dictionary)):
        data[descriptor_dictionary[descriptor]] = data['variable'].apply(lambda x: x.split('_')[idx])
    data['individual'] = data['monitor'] + '-' + data['variable'].apply(lambda x: x.split('_')[-1])
    data = data.drop(columns=['variable'])
    return data
        
data = interpret_layouts(monitors,layouts,descriptors)

#==============================================================================

#                      A N A L Y S E   E P O C H S  :

period = int(convert_time(period,'S')) # <- ensure period is in seconds    

def annotate_zeitgeber(dataframe):
    '''using a dataframe generated by interpret_layouts, map zt to timepoints.
    This will create a new column called 'zt'.'''
    dataframe = dataframe.copy()
    times_unq = dataframe['time'].unique() # <- list of all time bins present in the dataset
    if period != 86400:
        print('E R R O R : annotate_zeitgeber not supported for periods other than "24H"...') # <- different period lengths would require using something other than a unique list of times.
        return None
    if period % len(times_unq) != 0:
        print('E R R O R : period and time_window are incompatible...')
        return None
    if zt0 not in times_unq:
        print('E R R O R : unable to annotate zeitgeber time as zt0 not in "time". Check trimming.') # would have to get nearest zt to zt0 from list, work out what this zt is (e.g. zt0.5), then go from there.
        return None
    times_unq = np.roll(times_unq,-times_unq.tolist().index(zt0)) # <- circular permutation of times so that zt0 is first
    zt_unq = np.linspace(0,24,len(times_unq)+1)[:-1] # <- list of equivalent zts for all time bins present in the dataset
    map_unq = {i:j for i,j in zip(times_unq,zt_unq)} # <- dictionary used for mapping zts to corresponding times
    dataframe['zt'] = dataframe['time'].map(map_unq)
    return dataframe

data = annotate_zeitgeber(data)

def dt_to_zt(time):
    '''converts a datetime.time() object into a zeitgeber fractional time (i.e.
    a float)'''
    h = float(time.strftime('%H:%M:%S').split(':')[0])
    m = float(time.strftime('%H:%M:%S').split(':')[1]) / 60.0
    s = float(time.strftime('%H:%M:%S').split(':')[2]) / 3600.0
    zt = h + m + s
    return zt

epoch_window = 'smart' # <- start and length, in zt time, for epoch bounds. Can either be set manually with start and length seperated by a comma (e.g. 6.0, '12H' | this would start at ZT6, lasting 12H) or set to 'smart', where epochs will start at ZT0 and last 24hrs.

def annotate_epochs(dataframe):
    '''annotates each row with an epoch ID based on lower and upper bounds set
    in epoch_window. This will create a new column called 'epoch'.'''
    global epoch_window
    
    dataframe = dataframe.copy() 
    dt_unq = sorted(set(dataframe[['date','time']].apply(lambda x: dt.datetime.strptime(' '.join(x),'%d %b %y %H:%M:%S'),axis=1))) # <- get ordered list of unique datetimes
    zt_unq = [] # <- list of zt values corresponding to each datetime in dt_unq
    for i in dt_unq:
        zt_list = data.loc[data['time']==i.time().strftime('%H:%M:%S')]['zt'].tolist()
        if zt_list.count(zt_list[0]) != len(zt_list):
            print('E R R O R : annotate_epochs not supported for non-standard periods...') # <- different period lengths would require using something other than a unique list of times.
            return None
        else:
            zt_unq.append(zt_list[0])
    
    if epoch_window=='smart':
        epoch_window = 0.0, '24H' # <- (low bound ZT, length)
    elif type(epoch_window[0])!=float:
        print('E R R O R : start in epoch_window must be a floating point number (i.e. 12.0, not 12 or "ZT12"')
        return None
    elif type(epoch_window[1])!=str or epoch_window[1][-1] not in ['S','M','H']:
        print('E R R O R : length in epoch_window must be in the format "XY" where X is an integer and Y is S (sec), M (min), or H (hour) e.g. "12H"')
        return None
    if epoch_window[1][-1]=='H':
        length_dt = dt.timedelta(hours=int(epoch_window[1][:-1]), minutes=0, seconds=0)
    elif epoch_window[1][-1]=='M':
        length_dt = dt.timedelta(hours=0, minutes=int(epoch_window[1][:-1]), seconds=0)
    elif epoch_window[1][-1]=='S':
        length_dt = dt.timedelta(hours=0, minutes=0, seconds=int(epoch_window[1][:-1]))
    epoch_max = (dt.datetime(2000,1,1) + dt.timedelta(hours=epoch_window[0]) + length_dt).time() # <- end of epoch window bounds (2000,1,1 is just a placeholder year,month,day)
    zt_max = sorted(set(zt_unq))[sorted(set(zt_unq)).index(dt_to_zt(epoch_max))-1] # <- Convert epoch max to zt (i.e. float), and select compatible zt_max (because time binning may mean that the zt_max does not match epoch_max)
    zt_range1 = np.roll(sorted(set(zt_unq)),-sorted(set(zt_unq)).index(epoch_window[0])) # <- range of zt in epoch window, starting at low epoch bound
    zt_range2 = zt_range1.tolist()[:zt_range1.tolist().index(zt_max)+1] # <- range of zt in epoch window, up to high epoch bound
    zt_dict = {i:j for i,j in enumerate(zt_range2)} # <- dictionary where zt are ordered from min (low epoch window bound) to max (upper epoch window bound)
    
    epoch_table = pd.DataFrame(columns=['datetime','epoch'])
    current_epoch, current_zt = 0, zt_unq[0]
    for idx,i in enumerate(dt_unq): # <- scan ordered list of unique datetimes, changing epoch id every time a the zt of that epoch is 
        if zt_unq[idx] not in zt_dict.values():
            epoch_id = np.nan
        elif zt_unq[idx] != zt_dict[max(zt_dict)]:
            epoch_id = current_epoch
        elif zt_unq[idx] == zt_dict[max(zt_dict)]:
            epoch_id = current_epoch
            current_epoch += 1
        epoch_table.loc[idx,'datetime'], epoch_table.loc[idx,'epoch'] = i, epoch_id
    map_unq = dict(zip(epoch_table['datetime'], epoch_table['epoch'])) # <- dictionary used for mapping epochs to corresponding datetime indices
    dataframe['epoch'] = dataframe['index'].map(map_unq)
    return dataframe

data = annotate_epochs(data)

def annotate_peaks(dataframe,n_peaks=1):
    '''vial-max annotates the row with maximum activity for each vial, at each
    epoch. Because vial-max is suscpetible to random anomalous blips in
    activity, a seperate parameter called vial-peak annotates broader peaks. If
    more than one peak is expected (i.e. not Anopheles gambiae), set n_peaks to
    something other than one.'''
    dataframe = dataframe.copy()
    dataframe['vial-max'] = False # <- single timepoint with highest activity (in some cases, this is not swarm time)
    dataframe['vial-peak'] = False # <- find peak with additional criteria to above (peak is wider that one reading, i.e. are not just a blip); NOTE: prominence parameter in sg.find_peaks may be useful
    for i in dataframe['individual'].unique():
        for j in dataframe['epoch'].unique():
            temp = dataframe.loc[(dataframe['individual']==i) & (dataframe['epoch']==j)]
            
            pdx, prop = sg.find_peaks(temp['value'], width=1) # <- get list of indices and properties for peaks (at least as wide as one time point)
            pdx = [k for kdx,k in enumerate(pdx) if kdx not in np.where(prop['widths']==1)[0]] # <- from the list of indices above, drop any that are only as wide as one time point
            idx = temp.index[pdx] # <- convert list of pdx indices to actual indices
            idxmax = temp.loc[idx,'value'].sort_values()[-n_peaks:].index # <- get the indices of the maximal peak(s).
            dataframe.loc[idxmax,'vial-peak'] = True
            
            vmax = temp['value'].max()
            if len(temp.loc[temp['value']==vmax])!=len(temp): # <- ignore vials where there was no variation in activity (e.g. all 0)
                dataframe.loc[(dataframe['individual']==i) & (dataframe['epoch']==j) & (dataframe['value']==vmax),'vial-max'] = True
            else:
                pass
    return dataframe

data = annotate_peaks(data)

#==============================================================================

#               U S E F U L   F U N C T I O N S   E T C .

def drop_dead(dataframe, threshold=24):
    ''' remove dead individuals from dataset based on whether they have been
    inactive for threshold number of hours.'''
    dataframe = dataframe.copy()
    dataframe['ztb'] = dataframe['epoch'].apply(lambda x: x*24.0) + dataframe['zt'] # <- continuous zt
    for i in dataframe['individual'].unique():
        temp = dataframe.loc[(dataframe['individual']==i) & (dataframe['value']==0)]
        gaps = [[start, end] for start, end in zip(temp['ztb'], temp['ztb'][1:]) if start+1 <= end]
        edges = iter(temp['ztb'].tolist()[:1] + sum(gaps, []) + temp['ztb'].tolist()[-1:])
        gdx = [[start, end] for start, end in zip(temp.index.tolist(), temp.index.tolist()[1:]) if start+1 <= end]
        edx = iter(temp.index.tolist()[:1] + sum(gdx, []) + temp.index.tolist()[-1:])
        idx = list(zip(edx,edx)) # <- list of start and end indices of stretches of inactivity
        length = np.array([end - start for start, end in zip(edges, edges)]) # <- the length, in hours, of each of these stretches
        if max(length)==length[-1] or max(length)>=threshold:
            a, b = idx[np.argmax(length)][0], temp.index.max() # <- get the start index of the longest stretch of inactivity (if mosquitoes appear dead) as well as the final index
            dataframe.drop(range(a,b+1), inplace=True)
    return dataframe

# Trim the first and last epochs to remove incomplete data:
data = data.loc[data['epoch'].isin(range(data['epoch'].min()+1,data['epoch'].max()))]

def analyse_activity(dataframe, timestamp, accumulate='vial-peak'):
    global descriptors    
    columns = ['individual', 'before', 'after', 'ratio']
    columns = columns + [descriptors[i] for i in descriptors.keys()]
    activity = pd.DataFrame(columns=columns)
    for i in dataframe['individual'].unique():
        temp = dataframe.loc[dataframe['individual']==i]
        idx = temp.loc[temp['index']==timestamp].index.values[0]
        before, after = temp.loc[:idx-1,:], temp.loc[idx:,:]
        temp2 = pd.DataFrame({'individual': [i], \
                              'before': [before.loc[before[accumulate]==True]['value'].mean()], \
                              'after': [after.loc[after[accumulate]==True]['value'].mean()], \
                              'ratio': [after.loc[after[accumulate]==True]['value'].mean() / before.loc[before[accumulate]==True]['value'].mean()]})
        for j in descriptors.values():
            temp2[j] = before[j].values[0]
        activity = pd.concat([activity, temp2]).reset_index(drop=True)
    return activity


#def analyse_activity2(dataframe):
#    '''produce dictionary containing different stats, arrays, dfs'''
#    !!! NOTE: SLEEP/WAKE REQUIRES DATA TO BE IN AT LEAST 5MIN BINS !!!
#    total_activity_count = ACTIVITY ACROSS ALL EPOCH
#    wake_activity_count (INSOMNIAC - ACTIVITY COUNT PER WAKING MINUTE I.E. SLEEP-NORMALISED ACTIVITY)
#    rest_bout_length (INSOMNIAC)
#    normalised_activity = NEED TO KNOW WHEN DRUG ADDED OR TREATMENT CHANGED; CAN THEN NORMALISE EACH VIAL TO BEFORE/AFTER TREATMENT
#    period_length = TAU (FAASX RESULTS SUBMITTED TO CYCLEP)
#    sleep_pct = 5 MINS OF INACTIVITY = SLEEP; PCT CALCULATED EVERY HR (INSOMNIAC)

#==============================================================================

#                                 P L O T

def get_ymax(dataframe, estimator, x, y, hue=None):
    '''calculates a global ymax to use for plotting from all data by averaging
    across x values (column name in dataframe, e.g. 'time') for a given y value
    (column name in dataframe, e.g. 'value'). Choose one of the following
    estimators for the averaging: 'mean', 'median' or 'sum'. Optionally define
    a hue to calculate based on subset data.'''
    dataframe = dataframe.copy()
    avg = []
    for i in set(dataframe[x]):
        if hue==None:
            estimator_dict = {'mean':dataframe.loc[dataframe[x]==i][y].mean(),\
                              'median':dataframe.loc[dataframe[x]==i][y].median(),\
                              'sum':dataframe.loc[dataframe[x]==i][y].sum()}
            avg.append(estimator_dict[estimator])
        else:
            for j in set(dataframe[hue]):
                estimator_dict = {'mean':dataframe.loc[(dataframe[x]==i) & (dataframe[hue]==j)][y].mean(),\
                                  'median':dataframe.loc[(dataframe[x]==i) & (dataframe[hue]==j)][y].median(),\
                                  'sum':dataframe.loc[(dataframe[x]==i) & (dataframe[hue]==j)][y].sum()}
                avg.append(estimator_dict[estimator])
    ymax = max(avg) + max(avg) / 20
    return ymax

def fade(alpha,function='cube'):
    '''
    convert an alpha value, alpha (float between 0 and 1), based on a function. Allows
    for more aesthetically pleasing colour fades at dawn/dusk. Chose from 'cube',
    'quint' or 'sig' functions.
    '''
    if function=='cube':
        x = np.linspace(0,1,100)
        y = [i**3 for i in x]
    elif function=='quint':
        x = np.linspace(0,1,100)
        y = [i**5 for i in x]
    elif function=='sig':
        x = np.linspace(-1.5,1.5,100)
        y = np.sin(x)
    z = [(i - min(y)) / (max(y) - min(y)) for i in y]
    x2 = np.linspace(0,1,100)
    f = si.interp1d(x2,z)
    beta = float(f(alpha))
    return beta

#~~~~~~~~~~

# Plot environmental data (example):

pal = sb.color_palette('tab10')
sb.set_palette(pal)

fig, ax1 = mplp.subplots()
ax2 = ax1.twinx()
sb.lineplot(ax=ax1,data=data,x='index',y='light-lux',color='#fdbf6f')
sb.lineplot(ax=ax2,data=data,x='index',y='temperature',color='#e31a1c')
xlabels =  [i.split(' ')[0] for i in data['date'].unique()]
ax1.set_xticklabels(xlabels)

#~~~~~~~~~~

# Plot activity peak against time (example):

temp = data.loc[data['vial-max']==True]
ax = sb.stripplot(data=temp, x='epoch', y='zt', hue='genotype', dodge=True, jitter=0.2)

#~~~~~~~~~~

# Plot population averaged actograms (example):

data['genotype|drug'] = data['genotype'].copy() + ' | ' + data['drug'].copy()
colours = {'wt':'#666666', 'Gene 1':'#beaed4', 'Gene 2':'#7fc97f', 'Gene 3':'#fdc086'}
#data['ztb'] = data['epoch'].apply(lambda x: x*24.0) + data['zt'] # <- continuous zt
estimator_dict = {'mean':np.mean,'median':np.median}

hue = 'genotype|drug'
estimator = 'median'
drug = 'Drug A'

temp = data.loc[data['drug']==drug]
fig = mplp.figure(figsize=[8,6])
gs = mplg.GridSpec(len(temp[hue].unique()),1,hspace=0.01)
ax = {}
ymax = get_ymax(data, estimator, 'index', 'value', hue=hue)
for idx,i in enumerate(temp[hue].unique()):
    temp2 = temp.loc[temp[hue]==i]
    missing = set(np.arange(min(set(temp2['ztb'])),max(temp2['ztb'])+0.5,0.5)) - set(temp2['ztb']) # <- check for missing timepoints (e.g. where Trikinetics was unplugged)
    if missing != set():
        for j in missing:
            temp2.loc[temp2.index.max()+1,'ztb'] = j # <- add in a blank row with the missing zt (this will mean a tick will be plotted at that time point with no data instead of bunching up the timepoints surrounding the missing data)
    ax[i] = fig.add_subplot(gs[idx])
    sb.lineplot(ax=ax[i], data=temp2, x='ztb', y='value', errorbar=None, estimator=estimator_dict[estimator], \
                color='black', linewidth=0.8)
    temp3 = pd.pivot_table(temp2, values='value', index='ztb', aggfunc=estimator_dict[estimator], \
                           dropna=False, fill_value=0).reset_index()
    y1 = np.zeros(len(set(temp2['ztb']))) - 1
    ax[i].fill_between(x=temp3['ztb'], y1=y1, y2=temp3['value'], color=colours[i.split('|')[0].strip()])
    ax[i].set_ylim(ymax=ymax)
    ax[i].annotate(i,[ax[i].get_xlim()[1]-ax[i].get_xlim()[1]/15, ymax-ymax/4.5],ha='right')
    ax[i].set_ylabel('')
    ax[i].axvline(92, clip_on=False, c='0.3', ls=':')
    if idx!=len(temp[hue].unique())-1:
        sb.despine(ax=ax[i], bottom=True, trim=True)
        ax[i].tick_params(which='both', bottom=False, labelbottom='False')
        ax[i].set_xlabel('')
    else:
        sb.despine(ax=ax[i], trim=True)
        ax[i].set_xticks([24,36,48,60,72,84,96,108,120,132,144,156])
        ax[i].set_xticklabels(['0','','24','','48','','72','','96','','120',''])
        ax[i].set_xlabel('Hour')
    for j in set(temp2['ztb']):
        lux = temp2.loc[temp2['ztb']==j]['light-lux'].tolist()[0]
        if np.isnan(lux):
            ax[i].fill_between([j-0.5,j+0.5],[ymax,ymax],facecolor='none',\
                               linewidth=0,zorder=0) # '0.6' , #d9d9d9
        else:
            alpha = fade(1 - lux / data['light-lux'].max(),function='quint')
            ax[i].fill_between([j-0.5,j+0.5],[ymax,ymax],color='0.6',\
                               alpha=alpha,zorder=0)
mplp.annotate('Beam breaks',[0.01, 0.5], xycoords='figure fraction', rotation=90, \
              ha='left', va='center')
mplp.tight_layout()