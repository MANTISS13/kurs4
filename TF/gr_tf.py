
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy import stats

data=pd.read_csv("C:\\Work\\emotion_results_tf.csv",sep=';')

y=data['Эмоция']
x = [x for x in range(len(y))]

print(y)
emotions = list(set(y))  # Получаем уникальные значения эмоций
labels = ['Anger', 'Sad', 'Neutral', 'Happy', 'Surprise']

y_numeric = [labels.index(emotion) for emotion in y]  # Кодируем эмоции числами
y=y_numeric
print(y_numeric)
# Подсчет линии тренда
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(x)), y)

fig = go.Figure()

# Добавляем основные данные
fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Data'))

# Добавляем линию тренда
fig.add_trace(go.Scatter(x=x, y=[slope*xi + intercept for xi in range(len(x))], mode='lines', name='Trendline'))

frames=[]
for i in range(0, len(x)):
    frames.append(go.Frame(data=[go.Scatter(x=x[:i+1], y=(y[:i+1]))]))

fig.frames = frames   

fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  updatemenus=[dict(direction="left", x=0, xanchor="left", y=0,
                                    type="buttons", buttons=[dict(label="►", method="animate", args=[None, {"fromcurrent": True,"frame": {"duration": 50, "redraw": False},
                                                                                                               "mode": "immediate",
                                                                                                               "transition": {"duration": 50}}]),
                                                             dict(label="❚❚", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                                                               "mode": "immediate",
                                                                                                               "transition": {"duration": 0}}])])],
                  margin=dict(l=0, r=0, t=0, b=0))
fig.update_yaxes(tickvals=[0, 1, 2, 3, 4],
                  ticktext = labels)
fig.update_traces(marker=dict(size=15,
                              line=dict(width=2,
                                        color='Black')),
                  selector=dict(mode='lines+markers'))
fig.show()



