import pandas as pd
from matplotlib import pyplot as plt

from getmobility import get_mobility
from model import run_simulation


m_imported = get_mobility('https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv.csv', shift = True)
m_imported = (m_imported[0]-40, m_imported[1])


hospitalised_df = pd.read_csv("C:\\Users\\ASUS\\OneDrive - Alma Mater Studiorum Università di Bologna\\\Documents\\Università\\Magistrale - Bologna\\Primo anno\\Primo Semestre\\Physics of Complex Systems\\Ricoveri_Bologna_2022_08_14.csv")
hospitalised = hospitalised_df.H
intensive_care = hospitalised_df.loc[:,"T"]
hospitalised = hospitalised + intensive_care
time = hospitalised_df.tempo

dt = 1/24
t,s,e,i,h,a,r,tot = run_simulation(m=m_imported, days = 365, dt = dt, norm = True)

plt.figure("Simulation test")
plt.scatter(time - 44,hospitalised/886891, label = 'H real', s = 2, color= '#1f77b4')
plt.plot(t * dt, h, label = 'H simulation con shift', linewidth = 2, color='#ff7f0e')

# plt.plot(t * dt, s, label = 'S', linewidth = 2)
# plt.plot(t * dt, e, label = 'E', linewidth = 2)
# plt.plot(t * dt, i, label = 'I', linewidth = 2)
# plt.plot(t * dt, a, label = 'A', linewidth = 2)
# plt.plot(t * dt, r, label = 'R', linewidth = 2)

plt.legend()
plt.grid(True)

plt.ylim(bottom = 0, top = 0.0011)
plt.ylabel('Population Fraction')
plt.xlim([0, max(t * dt)])
plt.xlabel('Days since patient zero introduction')
plt.ylabel('People')

plt.show()


