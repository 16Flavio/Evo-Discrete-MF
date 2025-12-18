class ConfigAblation:
    def __init__(self):
        # --- INITIALISATION ---
        self.use_svd = True
        self.use_kmeans = True
        self.use_nmf = True
        self.use_greedy = True
        
        # --- CROSSOVER STRATEGY ---
        # "UNIFORM" = Mélange aléatoire des gènes (Classique)
        # "MEAN" = Moyenne des parents (Tentative de consensus)
        # "BOTH" = Utilisationn des stategies UNIFORM et MEAN
        self.crossover_type = "UNIFORM" 
        
        # --- RESTART STRATEGY ---
        # "FULL" = Smart Restart actuel (Soft -> Ruins -> Alien)
        # "SIMPLE" = Random Restart uniquement (Garder le meilleur + Random)
        self.restart_mode = "FULL" 
        
        # --- PHASES ---
        # Si True, l'algo peut transposer les matrices (DIRECT <-> TRANSPOSE)
        # Si False, il reste toujours en mode DIRECT
        self.allow_transpose = True

        # --- MUTATION ---
        # "SWAP" = Échange de deux gènes
        # "GREEDY" = Mutation gloutonne
        # "NOISE" = Ajout de bruit gaussien
        # "ALL" = Utilisation de toutes les stratégies de mutation
        self.mutation_type = "SWAP"

        # --- MODE ---
        # "IMF" = Mode de factorisation entière
        # "BMF" = Mode de factorisation binaire
        # "RELU" = Mode de factorisation avec ReLU
        self.factorization_mode = "IMF"

        # --- AFFICHAGE ---
        self.debug_mode = False  # Si True, affiche des informations de debug supplémentaires

    def __str__(self):
        return (f"Config(SVD={self.use_svd}, KMEANS={self.use_kmeans}, "
                f"CROSS={self.crossover_type}, RESTART={self.restart_mode}, "
                f"TRANSPOSE={self.allow_transpose}), MUTATION={self.mutation_type}, DEBUG={self.debug_mode})")