# -*- coding: utf-8 -*-
"""
Created on Tues May 12 16:40:00 2022
@author: anannugroho@mail.unnes.ac.id
"""
#%% INISIALISASI: Individu
from random import randint
def individu(JumlahGen, nilai_min, nilai_max):
    return [ randint(nilai_min,nilai_max) for x in range(JumlahGen) ]
""" Ex syntax: individu(5,0,10) 
    Artinya: individu terdiri 5 gen dengan rentang 0-10 """ 
#%% INISIALISASI: Populasi
def populasi(JumlahIndidu, JumlahGen, nilai_min, nilai_max):
    return [ individu(JumlahGen, nilai_min, nilai_max) for x in range(JumlahIndidu) ]

""" Ex syntax: populasi(3,5,0,10)
    Artinya: populasi terdiri atas 3 individu & masing2x individu 
    terdiri atas 5 gen dengan rentang antara 0-10 """ 
#%% EVALUASI: Fitness
from operator import add
from functools import reduce

def fitness(individu, target):
    JumlahGenTotal = reduce(add, individu, 0)
    return abs(target-JumlahGenTotal)

"""'Penentuan fungsi fitness berprinsip semakin mendekati nilai target
    maka susunan gen-gen dalam individu tersebut semakin bagus.
  * ttg fungsi "add" & "reduce" silakan cek di dokumentasi Python (doc.python.org)"""
#%% EVALUASI: Rerata Fitness
def reratafitness(populasi, target):
    JumlahTotal = reduce(add, (fitness(x, target) for x in populasi), 0)
    return JumlahTotal / (len(populasi) * 1.0)

""" Ex syntax:
    A = populasi (3,5,0,10)
    reratafitness(A,100) 
    Artinya: A adalah populasi yg trdiri dari 3 individu,
    masing2x 5 gen dengan rentang 0-10. Masing2x individu tsb punya nilai fitnnes
    yang kemudian kesemuanya dirata-ratakan dengan fungsi "reratafitness" tersebut"""
#%% SELEKSI: Orang_Tua dg Elitisme
# target = 300
# Populasi = populasi(7,5,0,100)
# Peluang_Dipertahankan = 0.3 # retain

# Fitness_n_Populasi = [(fitness(x, target),x) for x in Populasi] 
# Sorting_Fitness = [x[0] for x in sorted(Fitness_n_Populasi)]
# Sorting_Populasi = [x[1] for x in sorted(Fitness_n_Populasi)]
# Jumlah_Individu_Dipertahankan = int(len(Sorting_Populasi)*Peluang_Dipertahankan)
# print('Dari',len(Populasi),'individu dlm Populasi, dipilih Orang_Tua sejumlah',Jumlah_Individu_Dipertahankan,'yaitu:')
# Orang_Tua = Sorting_Populasi[:Jumlah_Individu_Dipertahankan]
# print(Orang_Tua)
# #%% SELEKSI: Orang_Tua dg Genetic Diversity
# """
# BAGAIMANA DENGAN INDIVIDU YG TERSISA?
# APAKAH TIDAK ADA KEMUNGKINAN MENJADI ORANG-TUA?
# Ini konsep Genetic Diversity
# """
# from random import random
# Peluang_Seleksi_Sisa_Individu = 0.5
# for Individu in Sorting_Populasi[Jumlah_Individu_Dipertahankan:]:
#     if Peluang_Seleksi_Sisa_Individu > random():
#         Orang_Tua.append(Individu)
# print('Seleksi Orang tua berdasar Elitisme & tambhan Genetic Diversity mjd',Orang_Tua)
# #%% REPRODUKSI: Mutasi Orang_Tua
# Peluang_Mutasi = 0.6
# print('Individu Orang_Tua pra Mutasi',Orang_Tua)
# n=0
# for Individu in Orang_Tua:
#         Acak = random()
#         print('Nilai Acak =', Acak)
#         if Peluang_Mutasi > Acak:     
#             print('Individu ke',n,'pada populasi Orang_Tua ini terjadi mutasi')
#             Urutan_Modifikasi_Gen = randint(0,len(Individu)-1)
#             print('yaitu urutan ke',Urutan_Modifikasi_Gen)
#             Individu[Urutan_Modifikasi_Gen] = randint(min(Individu), max(Individu))
#             n = n+1
#             print(Individu)
# print('Individu Orang_Tua Mutan',Orang_Tua)
# #%% REPRODUKSI: Perkawinan / Cross-Over
# Jumlah_Ortu = len(Orang_Tua)
# Kekurangan_Individu = len(Populasi)-Jumlah_Ortu
# Anak =[]
# while len(Anak) < Kekurangan_Individu:
#     No_Urut_Ayah = randint(0, Jumlah_Ortu-1)
#     No_Urut_Ibu = randint(0, Jumlah_Ortu-1)
#     if No_Urut_Ayah != No_Urut_Ibu:
#         Ayah = Orang_Tua[No_Urut_Ayah]
#         Ibu = Orang_Tua[No_Urut_Ibu]
#         Batas_Potong_Gen = round(len(Ayah)/2)
#         Keturunan = Ayah[:Batas_Potong_Gen]+Ibu[Batas_Potong_Gen:]
#         Anak.append(Keturunan)
# Orang_Tua.extend(Anak)
#%% EVOLUSI: Update Populasi Baru
from random import random
# Elitisme
def evolusi (Populasi, target, Peluang_Dipertahankan=0.2, Peluang_Seleksi_Sisa_Individu=0.05, Peluang_Mutasi=0.01):
    Fitness_n_Populasi = [(fitness(x, target),x) for x in Populasi]     
    Sorting_Populasi = [x[1] for x in sorted(Fitness_n_Populasi)]
    Jumlah_Individu_Dipertahankan = int(len(Sorting_Populasi)*Peluang_Dipertahankan)
    Orang_Tua = Sorting_Populasi[:Jumlah_Individu_Dipertahankan]
    
    # Genetic Diversity
    for Individu in Sorting_Populasi[Jumlah_Individu_Dipertahankan:]:
        if Peluang_Seleksi_Sisa_Individu > random():
            Orang_Tua.append(Individu)
            
    # Mutasi
    for Individu in Orang_Tua:    
        if Peluang_Mutasi > random():     
           Urutan_Modifikasi_Gen = randint(0,len(Individu)-1)
           Individu[Urutan_Modifikasi_Gen] = randint(min(Individu), max(Individu))
           
    # Cross_Over
    Jumlah_Ortu = len(Orang_Tua)
    Kekurangan_Individu = len(Populasi)-Jumlah_Ortu
    Anak =[]
    while len(Anak) < Kekurangan_Individu:
        No_Urut_Ayah = randint(0, Jumlah_Ortu-1)
        No_Urut_Ibu = randint(0, Jumlah_Ortu-1)
        if No_Urut_Ayah != No_Urut_Ibu:
            Ayah = Orang_Tua[No_Urut_Ayah]
            Ibu = Orang_Tua[No_Urut_Ibu]
            Batas_Potong_Gen = round(len(Ayah)/2)
            Keturunan = Ayah[:Batas_Potong_Gen]+Ibu[Batas_Potong_Gen:]
            Anak.append(Keturunan)
    Orang_Tua.extend(Anak)
    return Orang_Tua
#%%   TRIAL = command
"""
Mencari 10 alternatif solusi untuk penjumlahan 5 angka rentang 0 s.d 100
yang meghasilkan nilai 300
"""
JumlahIndividu = 11
target = 300
JumlahGen = 5
nilai_min = 0
nilai_max = 100
# Peluang_Dipertahankan=0.2
# Peluang_Seleksi_Sisa_Individu=0.05
# Peluang_Mutasi=0.01

Populasi = populasi(JumlahIndividu, JumlahGen, nilai_min, nilai_max)
rekap_fitness = [reratafitness(Populasi, target),]
print("Rekap fitness awal",rekap_fitness)

for i in range(1000):
    Populasi = evolusi(Populasi, target)
    rekap_fitness.append(reratafitness(Populasi, target))
for data in rekap_fitness:
    print(data)