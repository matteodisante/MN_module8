import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from math_utils import circular_mi_theoretical, ordered_wienman_exponential_mi_theoretical, gamma_exponential_mi_theoretical, correlated_gaussian_rv_mi_theoretical



if __name__ == '__main__':
    print("Scegli una delle seguenti distribuzioni per calcolare la MI teorica:")
    print("1. bivariate_normal")
    print("2. circular")
    print("3. weinman_exponential")
    print("4. gamma_exponential")
    
    # Ciclo per gestire l'errore in caso di scelta non valida
    distribuzioni_valide = {"bivariate_normal", "circular", "weinman_exponential", "gamma_exponential"}
    while True:
        scelta = input("Inserisci il nome della distribuzione: ").strip().lower()
        if scelta in distribuzioni_valide:
            break
        else:
            print("Distribuzione non riconosciuta. Riprova.\n")
    
    try:
        if scelta == "bivariate_normal":
            r = float(input("Inserisci il valore del parametro di correlazione r (tra -1 e 1): "))
            mi = correlated_gaussian_rv_mi_theoretical(r)
        elif scelta == "circular":
            a = float(input("Inserisci il valore del raggio interno a: "))
            b = float(input("Inserisci il valore del raggio esterno b: "))
            c = float(input("Inserisci il valore del raggio intermedio c: "))
            mi = circular_mi_theoretical(a, b, c)
        elif scelta == "weinman_exponential":
            u = float(input("Inserisci il valore del parametro u (u>0): "))
            mi = ordered_wienman_exponential_mi_theoretical(u)
        elif scelta == "gamma_exponential":
            u = float(input("Inserisci il valore del parametro u (u > 0): "))
            mi = gamma_exponential_mi_theoretical(u)
        
        print(f"\nLa Mutual Information teorica è: {mi}")
    
    except Exception as e:
        print(f"Si è verificato un errore durante il calcolo: {e}")