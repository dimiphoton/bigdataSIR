def apply_interventions(model, interventions):
    """
    Applique des interventions au modèle SEIR.

    Paramètres :
    - model : Une instance de la classe SEIRModel.
    - interventions : Un dictionnaire contenant les paramètres d'intervention.

    Retourne :
    - model : L'instance SEIRModel mise à jour avec les interventions appliquées.
    """
    # Ajuster les paramètres du modèle en fonction des interventions
    model.beta *= (1 - interventions.get('mask_wearing', 0))
    model.p_sym *= (1 - interventions.get('symptomatic_isolation', 0))
    model.beta *= (1 - interventions.get('lockdown', 0))
    return model
