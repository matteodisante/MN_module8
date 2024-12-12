import numpy as np



if __name__ == "__main__":
    # Parametri iniziali
    x0, y0 = 2, 2  # Punto iniziale
    steps = int(1e6)  # Numero di passi totali
    delta = 1 # Raggio massimo della proposta uniforme
    n_cores = 8  # Numero di core da utilizzare
    burn_in = int(1e4)  # Numero di punti da scartare per ogni worker
    
    # Scegli la distribuzione
    distribution_choice = "gaussian"  
    target_distribution = distribution_functions[distribution_choice]
    
    if distribution_choice in ["gamma_exponential"]:
        check_value = True
    if distribution_choice in ["gaussian"]:
        check_value = False
    


    # Genera punti usando Metropolis-Hastings in parallelo
    samples = run_parallel_metropolis_hastings(
        target_distribution, 
        proposal_sampler, 
        proposal_probability, 
        x0, y0, steps, delta, n_cores, burn_in, check = check_value
    )
    
    corr = np.corrcoef(samples[:,0], samples[:,1])
    print(rf"Corr(X_{1}, X_{2}) = {corr}")
    
    print(f"End of samples generation \n Number of samples: {samples.shape[0]}")
    
      
    # Istogramma 2D dai campioni generati
    hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=60, density=True)
    
    
    
    # Creazione della distribuzione teorica in 3D
    x = np.linspace(xedges.min(), xedges.max(), 300)
    y = np.linspace(yedges.min(), yedges.max(), 300)
    xgrid, ygrid = np.meshgrid(x, y)
    
    # Calcolo della densità teorica su ogni punto in modo vettoriale
    #density = np.zeros_like(xgrid)  # Matrice vuota per la densità
    # Funzione vettorializzata di target_distribution
    #target_distribution_vec = np.vectorize(target_distribution)
    # Calcolo della densità teorica 
    # Calcolo della densità teorica direttamente
    density = target_distribution(xgrid, ygrid)
    
        
    # Proiezione lungo X e Y per ottenere le marginali dai dati
    marginal_x_data = hist.sum(axis=0) * (yedges[1] - yedges[0])  # Marginale lungo X
    marginal_y_data = hist.sum(axis=1) * (xedges[1] - xedges[0])  # Marginale lungo Y
    
    # Marginali teoriche dalla PDF congiunta
    marginal_x_theoretical = density.sum(axis=0) * (y[1] - y[0])
    marginal_y_theoretical = density.sum(axis=1) * (x[1] - x[0])
    
    # Grafico scatter 2D delle coppie estratte
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.title("Scatter plot delle coppie estratte (Gaussiana bivariata con Metropolis-Hastings)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show(block=False)
    
    
    # Grafico 3D dell'istogramma dei dati generati
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = np.ones_like(zpos) * (xedges[1] - xedges[0])
    dz = hist.ravel()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
    ax.set_title("Istogramma 3D dei punti generati (Gaussiana bivariata con MCMC)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Densità")
    plt.show(block=False)
    
    
    # Grafico 3D della PDF congiunta teorica
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xgrid, ygrid, density, cmap="viridis", edgecolor="none", alpha=0.8)
    ax.set_title("Distribuzione teorica 3D (Gaussiana bivariata)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Densità")
    plt.show(block=False)
    

    
    # Istogrammi delle marginali dai dati
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(samples[:, 0], bins=50, density=False, alpha=0.7, label="Marginale X (dati)")
    plt.title("Marginale lungo X (istogramma)")
    plt.xlabel("X")
    plt.ylabel("Densità")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(samples[:, 1], bins=50, density=False, alpha=0.7, label="Marginale Y (dati)")
    plt.title("Marginale lungo Y (istogramma)")
    plt.xlabel("Y")
    plt.ylabel("Densità")
    plt.grid(True)
    plt.show(block=False)
    
    
    
    # Grafico 2D delle marginali teoriche
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, marginal_x_theoretical, label="Marginale X (teorica)")
    plt.title("Marginale lungo X (teorica)")
    plt.xlabel("X")
    plt.ylabel("Densità")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(y, marginal_y_theoretical, label="Marginale Y (teorica)")
    plt.title("Marginale lungo Y (teorica)")
    plt.xlabel("Y")
    plt.ylabel("Densità")
    plt.grid(True)
    plt.show(block=False)
    
    
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, marginal_x_theoretical, label="Marginale X (teorica)")
    plt.title("Marginale lungo X (teorica)")
    plt.xlabel("X")
    plt.ylabel("Densità")
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(y, marginal_y_theoretical, label="Marginale Y (teorica)")
    plt.title("Marginale lungo Y (teorica)")
    plt.xlabel("Y")
    plt.ylabel("Densità")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show(block=False)
    
        
    
    # Grafico 2D degli isotgrammi in scala semilogy o log-log
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(samples[:, 0], bins=100, density=False, alpha=0.7, label="Marginale X (dati)")
    plt.title("Marginale lungo X (istogramma)")
    plt.xlabel("X")
    plt.ylabel("conteggi")
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(samples[:, 1], bins=100, density=False, alpha=0.7, label="Marginale Y (dati)")
    plt.title("Marginale lungo Y (istogramma)")
    plt.xlabel("Y")
    plt.ylabel("conteggi")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
