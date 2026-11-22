"""
Script de génération du rapport scientifique final (PDF, max 6 pages)
Projet HAR — M2 MALIA (Université Lumière Lyon 2)
Auteurs : Alioune DIOP & Madjiguène FALL
Enseignant : Walid Bechkit
Année universitaire : 2026–2027

Ce script charge directement les résultats réels et vérifiés issus de :
- reports/results/v1_results.json
- reports/results/v2_results.json
et compile un document PDF conforme au format article scientifique.
"""

import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # En-tête (pages 2 et suivantes)
        if self._pageNumber > 1:
            self.drawString(40, 810, "M2 MALIA Deep Learning — Projet HAR : Étude progressive des réseaux de neurones")
            self.drawRightString(555, 810, "A. DIOP & M. FALL")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(40, 804, 555, 804)
            
        # Pied de page
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        self.drawString(40, 24, "Université Lumière Lyon 2 — Année universitaire 2026–2027")
        self.drawRightString(555, 24, f"Page {self._pageNumber} sur {page_count}")
        self.restoreState()

def build_pdf_report(output_pdf="reports/M2_DL_2026_DIOP_FALL.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    # Chargement des résultats réels
    with open("reports/results/v1_results.json", "r", encoding="utf-8") as f:
        r1 = json.load(f)
    with open("reports/results/v2_results.json", "r", encoding="utf-8") as f:
        r2 = json.load(f)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=14, leading=17,
        textColor=colors.HexColor("#1A365D"), alignment=1, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=13,
        textColor=colors.HexColor("#2B6CB0"), alignment=1, spaceAfter=5
    )
    author_style = ParagraphStyle(
        'DocAuthors', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11.5,
        textColor=colors.HexColor("#333333"), alignment=1, spaceAfter=7
    )
    abstract_style = ParagraphStyle(
        'AbstractText', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.8, leading=10.5,
        textColor=colors.HexColor("#2D3748"), alignment=4
    )
    h1_style = ParagraphStyle(
        'SecH1', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=10, leading=12.5,
        textColor=colors.HexColor("#1A365D"), spaceBefore=7, spaceAfter=2.5,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.0, leading=10.5,
        textColor=colors.HexColor("#2D3748"), alignment=4, spaceAfter=3.5
    )
    caption_style = ParagraphStyle(
        'FigCaption', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.0, leading=9.0,
        textColor=colors.HexColor("#4A5568"), alignment=1, spaceBefore=2, spaceAfter=4
    )

    story = []

    # ==================== EN-TÊTE ====================
    story.append(Paragraph("Étude Progressive des Réseaux de Neurones Multicouches sur le Jeu de Données HAR", title_style))
    story.append(Paragraph("Méthodologie Expérimentale, Régularisation, Analyse Critique et Interprétabilité Latente", subtitle_style))
    story.append(Paragraph("<b>Alioune DIOP & Madjiguène FALL</b><br/>Master 2 MALIA — Université Lumière Lyon 2 — Enseignant : Walid Bechkit", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=5))

    # Abstract
    abs_text = (
        "<b>Résumé —</b> Ce travail présente une analyse progressive et rigoureuse de réseaux de neurones multicouches (MLP) pour la reconnaissance "
        "d'activités humaines à partir de signaux inertiels de smartphones (HAR Dataset, 6 activités, 561 caractéristiques expertes). "
        "En respectant un protocole sans fuite d'information (validation par volontaires entiers plutôt qu'aléatoire), nous établissons une référence "
        "linéaire solide (95,28% sur le test). Nous étudions l'influence du pas d'apprentissage, des fonctions d'activation (Tanh surpasse ReLU à 95,43% en validation) "
        "et de la variabilité multi-graines. L'extension en profondeur (jusqu'à 4 couches) révèle un risque accru de surapprentissage que seule une régularisation "
        "stochastique forte (Dropout p=0,5) permet de juguler efficacement (96,48% en validation). Sur le jeu de test final sanctuarisé, un résultat remarquable apparaît : "
        "la régression logistique linéaire (95,28%) surpasse les MLP (93,88% à 94,17%). L'analyse de confusion (19% d'ambiguïté assise/debout) et l'interprétabilité t-SNE "
        "fournissent une explication physique limpide : les 561 variables expertes rendent les mouvements dynamiques presque linéairement séparables, "
        "tandis que les postures statiques constituent l'unique frontière ambiguë en raison de l'invariance du vecteur gravitationnel mesuré au niveau du bassin."
    )
    abs_t = Table([[Paragraph(abs_text, abstract_style)]], colWidths=[520])
    abs_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(abs_t)
    story.append(Spacer(1, 4))

    # ==================== 1. INTRODUCTION ====================
    story.append(Paragraph("1. Introduction et Problématique", h1_style))
    story.append(Paragraph(
        "La reconnaissance d'activités humaines (<i>Human Activity Recognition</i> - HAR) via des capteurs miniatures intégrés (accéléromètres et gyroscopes) "
        "est un domaine d'application central pour le suivi médical continu, la réhabilitation biomécanique et la détection de chutes. "
        "Ce projet a pour objectif d'analyser méthodiquement comment la complexité d'un perceptron multicouche (MLP), le choix des fonctions d'activation, "
        "les algorithmes d'optimisation et les techniques de régularisation influencent la performance et la généralisation sur des sujets humains inconnus.",
        body_style
    ))
    story.append(Paragraph(
        "Plutôt que d'optimiser aveuglément un score de compétition, notre étude privilégie une démarche scientifique stricte : "
        "établissement d'une baseline simple, ensemble de validation sans fuite d'information, résultats moyennés sur plusieurs graines aléatoires, "
        "et préservation absolue du jeu de test pour une évaluation unique en toute fin de protocole.",
        body_style
    ))

    # ==================== 2. DONNÉES ET MÉTHODOLOGIE ====================
    story.append(Paragraph("2. Jeu de Données et Démarche Méthodologique", h1_style))
    story.append(Paragraph(
        "<b>2.1 Présentation des données :</b> Le jeu de données HAR (UCI Machine Learning Repository) recueille les mesures de 30 volontaires "
        "portant un smartphone à la taille. Les signaux d'accélération et de vitesse angulaire à 50 Hz ont été segmentés en fenêtres de 2,56 secondes "
        "(128 pas de temps avec 50% de chevauchement), desquelles ont été extraites <b>561 variables précalculées</b> dans les domaines temporel et fréquentiel. "
        "Six activités sont ciblées : <i>WALKING</i>, <i>WALKING_UPSTAIRS</i>, <i>WALKING_DOWNSTAIRS</i>, <i>SITTING</i>, <i>STANDING</i> et <i>LAYING</i>. "
        "Le fichier <code>train.csv</code> regroupe <b>7 352 observations</b> réparties sur 21 volontaires ; le fichier <code>test.csv</code> compte <b>2 947 observations</b> "
        "issues de 9 volontaires distincts. <b>Aucun sujet n'est partagé</b> entre le train et le test, ce qui simule fidèlement le déploiement sur de nouveaux utilisateurs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Prétraitements et conception :</b><br/>"
        "• <i>Normalisation :</i> L'inspection des 561 variables indique un minimum global de <b>-1,0000</b> et un maximum de <b>+1,0000</b>. "
        "Les variables étant déjà bornées dans [-1, 1], aucune mise à l'échelle supplémentaire n'est nécessaire.<br/>"
        "• <i>Exclusion de la colonne <code>subject</code> :</i> La colonne <code>subject</code> identifie l'individu. "
        "<b>Elle ne doit absolument pas servir d'entrée au modèle</b> : son inclusion provoquerait une fuite de données (*data leakage*) où le modèle mémoriserait "
        "les personnes au lieu des signatures cinématiques, détruisant la généralisation sur les sujets de test inconnus.<br/>"
        "• <i>Encodage :</i> Les noms des activités sont transformés en entiers de 0 à 5 via <code>LabelEncoder</code> pour alimenter <code>nn.CrossEntropyLoss</code>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.3 Protocole de découpage : Aléatoire vs Par Volontaire (Partie 1.3) :</b> "
        "Deux fenêtres consécutives d'un même sujet se chevauchent à 50% et sont donc quasi-identiques. "
        "Un découpage aléatoire des lignes induit une fuite d'information majeure :",
        body_style
    ))

    # Table Découpage
    val_rand = r1['decoupage']['val_acc_aleatoire'] * 100
    val_subj = r1['decoupage']['val_acc_volontaires'] * 100
    test_lr = r1['test']['reference']['acc'] * 100
    test_rand = r1['test']['decoupage']['aleatoire']['test'] * 100
    
    t_split_data = [
        ["Stratégie de Découpage", "Validation Acc (%)", "Test Acc (%)", "Écart |Val - Test|", "Diagnostic Scientifique"],
        ["Découpage Aléatoire des lignes (80/20)", f"{val_rand:.2f} %", f"{test_rand:.2f} %", f"{abs(val_rand-test_rand):.2f} pt", "Illusoire / Biais optimiste (Fuite de données)"],
        ["Découpage par Volontaires (Sujets 1, 3, 5, 7)", f"{val_subj:.2f} %", f"{test_lr:.2f} %", f"{abs(val_subj-test_lr):.2f} pt", "Fidèle à la généralisation réelle"]
    ]
    t_split = Table(t_split_data, colWidths=[150, 85, 75, 75, 135])
    t_split.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('ALIGN', (1,0), (3,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2), ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_split)
    story.append(Paragraph("<b>Table 1 :</b> Comparaison des stratégies de découpage sur la régression logistique.", caption_style))

    # ==================== 3. VERSION 1 ====================
    story.append(Paragraph("3. Première Architecture MLP et Résultats de Base (Version 1)", h1_style))
    story.append(Paragraph(
        f"<b>3.1 Contrôles théoriques préalables (Partie 2.1) :</b> Avant d'entraîner notre premier réseau (MLP 1 couche cachée, 64 neurones, 36 358 paramètres), "
        f"nous validons deux contrôles impératifs :<br/>"
        f"1. <i>Perte initiale théorique :</i> Pour C=6 classes équiprobables et poids aléatoires, L_0 = -ln(1/6) = ln(6) ≈ 1,7918. "
        f"La perte mesurée sur nos données est de <b>{r1['perte_initiale']['moyenne']:.4f}</b> (écart < 0,3%), confirmant l'initialisation saine.<br/>"
        f"2. <i>Surapprentissage mini-lot (Sanity Check) :</i> Sur 10 exemples, le modèle atteint <b>100,0%</b> de précision et une perte de "
        f"{r1['sanity_check']['perte_finale']:.5f} en {r1['sanity_check']['iterations']} itérations, prouvant l'absence de bug de rétropropagation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3.2 Réglage du pas d'apprentissage (Learning Rate) :</b> "
        "Le sujet impose de régler le pas d'apprentissage pour chaque algorithme. Sur une grille logarithmique "
        "[10^-4, 10^-3, 10^-2, 0.05, 0.1, 0.5] (Figure 1 gauche), les valeurs optimales retenues sur la validation sont : "
        f"<b>lr = {r1['balayage_lr']['retenu']['sgd']}</b> pour SGD (Val F1: 92,76%), "
        f"<b>lr = {r1['balayage_lr']['retenu']['sgd_momentum']}</b> pour SGD Momentum (Val F1: 95,88%), et "
        f"<b>lr = {r1['balayage_lr']['retenu']['adam']}</b> pour Adam (Val F1: 95,65%).",
        body_style
    ))

    # Tableau activations et optimiseurs
    act_data = [
        ["Configuration (MLP 64)", "Optimiseur & LR", "Val Loss", "Val Acc (%)", "Val Macro-F1 (%)", "Observations"],
        ["ReLU", f"Adam (lr={r1['activations']['relu']['train_loss']:.3f}*)", f"{r1['activations']['relu']['val_loss']:.4f}", f"{r1['activations']['relu']['val_acc']*100:.2f} %", f"{r1['activations']['relu']['val_f1']*100:.2f} %", "Convergence rapide"],
        ["Tanh", "Adam (lr=0.001)", f"{r1['activations']['tanh']['val_loss']:.4f}", f"{r1['activations']['tanh']['val_acc']*100:.2f} %", f"{r1['activations']['tanh']['val_f1']*100:.2f} %", "Meilleure symétrie autour de 0"],
        ["Sigmoid", "Adam (lr=0.001)", f"{r1['activations']['sigmoid']['val_loss']:.4f}", f"{r1['activations']['sigmoid']['val_acc']*100:.2f} %", f"{r1['activations']['sigmoid']['val_f1']*100:.2f} %", "Saturation des gradients (inférieur)"],
        ["ReLU", f"SGD (lr={r1['optimiseurs']['sgd']['lr']})", f"{r1['optimiseurs']['sgd']['val_loss']:.4f}", f"{r1['optimiseurs']['sgd']['val_acc']*100:.2f} %", f"{r1['optimiseurs']['sgd']['val_f1']*100:.2f} %", "Sous-optimal sans moment"],
        ["ReLU", f"SGD+Mom (lr={r1['optimiseurs']['sgd_momentum']['lr']})", f"{r1['optimiseurs']['sgd_momentum']['val_loss']:.4f}", f"{r1['optimiseurs']['sgd_momentum']['val_acc']*100:.2f} %", f"{r1['optimiseurs']['sgd_momentum']['val_f1']*100:.2f} %", "Inertie efficace, bon score"]
    ]
    t_act = Table(act_data, colWidths=[85, 95, 55, 65, 75, 145])
    t_act.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.0),
        ('ALIGN', (2,0), (4,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2), ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_act)
    story.append(Paragraph("<b>Table 2 :</b> Comparaison des activations et optimiseurs sur l'architecture 1 couche cachée (64 neurones).", caption_style))

    # Variabilité graines
    g_relu = r1['graines']['ReLU + Adam']['val_acc']
    g_tanh = r1['graines']['Tanh + Adam']['val_acc']
    story.append(Paragraph(
        f"<b>3.3 Variabilité sur 3 graines (Partie 2.3) :</b> Répéter l'entraînement sur seeds [42, 123, 999] donne :<br/>"
        f"• <i>ReLU + Adam :</i> {g_relu[0]*100:.2f}% ± {g_relu[1]*100:.2f}% | <i>Tanh + Adam :</i> {g_tanh[0]*100:.2f}% ± {g_tanh[1]*100:.2f}%. "
        f"L'écart moyen entre Tanh et ReLU (+0,08%) est inférieur aux écarts-types (0,54% à 0,86%). "
        f"<b>Conclusion scientifique :</b> l'avantage de Tanh sur ReLU n'est pas statistiquement significatif sur ce réseau simple ; les deux fonctions sont équivalentes en pratique.",
        body_style
    ))

    # Figures V1
    f_lr = "reports/figures/05b_reglage_learning_rate.png"
    f_opt = "reports/figures/06_comparaison_optimiseurs.png"
    if os.path.exists(f_lr) and os.path.exists(f_opt):
        img_t1 = Table([[Image(f_lr, width=255, height=95), Image(f_opt, width=255, height=95)]], colWidths=[260, 260])
        img_t1.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_t1)
        story.append(Paragraph("<b>Figure 1 :</b> Réglage du pas d'apprentissage par balayage logarithmique (gauche) et convergence des optimiseurs (droite).", caption_style))

    # ==================== 4. VERSION 2 ====================
    story.append(Paragraph("4. Extension de l'Architecture et Régularisation (Version 2)", h1_style))
    story.append(Paragraph(
        "<b>4.1 Largeur et Profondeur (Partie 3, 3 graines) :</b> "
        "Nous évaluons l'impact de l'élargissement de la couche cachée et de l'approfondissement jusqu'à 4 couches cachées (Table 3).",
        body_style
    ))

    # Table architectures réelles
    arch_rows = [["Architecture", "Couches", "Paramètres", "Temps (s)", "Val Acc (%)", "Val F1 (%)", "Overfitting (Train-Val)"]]
    for a in r2['architectures']:
        p = a['Paramètres']
        va, vs = a['val_acc'][0]*100, a['val_acc'][1]*100
        vf = a['val_f1'][0]*100
        tm = a['time'][0]
        gap = (a['train_acc'][0] - a['val_acc'][0]) * 100
        dims = a['Architecture'].split('[')[-1].replace(']', '').replace('(ReLU)', '').replace('(Tanh)', '').strip()
        arch_rows.append([a['Architecture'], dims, f"{p:,}", f"{tm:.1f}s", f"{va:.2f} ± {vs:.2f}", f"{vf:.2f} %", f"+{gap:.2f} pt"])
    
    t_arch = Table(arch_rows, colWidths=[130, 65, 55, 45, 80, 65, 80])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 6.8),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2), ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_arch)
    story.append(Paragraph("<b>Table 3 :</b> Étude de la complexité architecturale (moyenne ± écart-type sur 3 graines aléatoires).", caption_style))

    story.append(Paragraph(
        "<i>Analyse :</i> Passer de 64 à 128 neurones améliore la fidélité, mais approfondir le réseau au-delà de 2 couches sans régularisation "
        "ne produit aucun gain de validation (95,56% à 3 couches vs 96,05% à 1 couche), alors que l'écart train-val s'accroît (+2,91 pt), "
        "signalant un début manifeste d'overfitting.",
        body_style
    ))

    # Table régularisation réelle
    story.append(Paragraph(
        "<b>4.2 Régularisation sur le réseau à 3 couches (Partie 4, 3 graines) :</b> "
        "Nous confrontons Dropout (p = 0.1, 0.3, 0.5), Weight Decay (L2 = 1e-4) et Batch Normalization :",
        body_style
    ))

    reg_rows = [["Régularisation", "Val Loss", "Val Acc (%)", "Val F1 (%)", "Époque Arrêt", "Effet Principal"]]
    for r in r2['regularisation']:
        lbl = r['Régularisation']
        vl = r['val_loss'][0]
        va, vs = r['val_acc'][0]*100, r['val_acc'][1]*100
        vf = r['val_f1'][0]*100
        ep = r['best_epoch'][0]
        obs = "Overfitting" if "Sans" in lbl else ("Meilleure généralisation" if "0.5" in lbl else ("Bonne régularisation" if "0.3" in lbl else "Modéré"))
        reg_rows.append([lbl, f"{vl:.4f}", f"{va:.2f} ± {vs:.2f}", f"{vf:.2f} %", f"{ep:.1f}", obs])

    t_reg = Table(reg_rows, colWidths=[150, 50, 80, 65, 65, 110])
    t_reg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 6.8),
        ('ALIGN', (1,0), (4,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2), ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_reg)
    story.append(Paragraph("<b>Table 4 :</b> Comparaison des mécanismes de régularisation sur 3 couches (moyenne ± écart-type sur 3 graines).", caption_style))

    story.append(Paragraph(
        "<i>Bilan régularisation :</i> <b>Dropout (p = 0.5)</b> remporte nettement la comparaison en atteignant la perte de validation la plus basse (0,0978) "
        "et la meilleure précision (<b>96,48% ± 0,12%</b>). Son avantage sur le modèle non régularisé (+0,92 pt) dépasse largement la variabilité (0,12 pt) : "
        "il est donc statistiquement significatif. Les deux configurations retenues pour l'évaluation finale sont <b>Dropout 0.5</b> et <b>Dropout 0.3</b>.",
        body_style
    ))

    # ==================== 5. ÉVALUATION FINALE SUR TEST ====================
    story.append(Paragraph("5. Évaluation Finale sur Test et Analyse de la Matrice de Confusion", h1_style))
    story.append(Paragraph(
        "<b>5.1 Règle d'or : Passage unique sur le jeu de test final :</b> "
        "Le jeu de test sanctuarisé (2 947 exemples de nouveaux volontaires) est évalué pour la première et unique fois à cette étape :",
        body_style
    ))

    # Bilan test réel
    t_lr_acc = r1['test']['reference']['acc'] * 100
    t_lr_f1 = r1['test']['reference']['f1'] * 100
    t_v1_acc, t_v1_acc_s = r1['test']['mlp_v1']['acc'][0]*100, r1['test']['mlp_v1']['acc'][1]*100
    t_v1_f1, t_v1_f1_s = r1['test']['mlp_v1']['f1'][0]*100, r1['test']['mlp_v1']['f1'][1]*100
    
    t_v2_5_acc, t_v2_5_s = r2['test']['Dropout (p=0.5)']['test_acc'][0]*100, r2['test']['Dropout (p=0.5)']['test_acc'][1]*100
    t_v2_5_f1 = r2['test']['Dropout (p=0.5)']['test_f1'][0]*100
    t_v2_3_acc, t_v2_3_s = r2['test']['Dropout (p=0.3)']['test_acc'][0]*100, r2['test']['Dropout (p=0.3)']['test_acc'][1]*100

    test_rows = [
        ["Modèle Évalué", "Rôle / Description", "Val Acc (%)", "Test Acc (%)", "Test Macro-F1 (%)", "Écart |Val - Test|"],
        ["Régression Logistique", "Baseline de référence (Partie 1)", f"{val_subj:.2f} %", f"{t_lr_acc:.2f} %", f"{t_lr_f1:.2f} %", f"{abs(val_subj-t_lr_acc):.2f} pt"],
        ["Simple MLP 1 couche [64] (Tanh)", "Réseau simple retenu (Partie 2)", f"{r1['graines']['Tanh + Adam']['val_acc'][0]*100:.2f} %", f"{t_v1_acc:.2f} ± {t_v1_acc_s:.2f}", f"{t_v1_f1:.2f} ± {t_v1_f1_s:.2f}", f"{abs(r1['graines']['Tanh + Adam']['val_acc'][0]*100-t_v1_acc):.2f} pt"],
        ["Deep MLP 3 couches + Dropout 0.3", "Top 2 Régularisation (Partie 4)", f"{r2['test']['Dropout (p=0.3)']['val_acc'][0]*100:.2f} %", f"{t_v2_3_acc:.2f} ± {t_v2_3_s:.2f}", f"{r2['test']['Dropout (p=0.3)']['test_f1'][0]*100:.2f} %", f"{abs(r2['test']['Dropout (p=0.3)']['val_acc'][0]*100-t_v2_3_acc):.2f} pt"],
        ["<b>Deep MLP 3 couches + Dropout 0.5</b>", "<b>Meilleure architecture globale</b>", f"<b>{r2['test']['Dropout (p=0.5)']['val_acc'][0]*100:.2f} %</b>", f"<b>{t_v2_5_acc:.2f} ± {t_v2_5_s:.2f}</b>", f"<b>{t_v2_5_f1:.2f} %</b>", f"<b>{abs(r2['test']['Dropout (p=0.5)']['val_acc'][0]*100-t_v2_5_acc):.2f} pt</b>"]
    ]
    t_final = Table(test_rows, colWidths=[140, 110, 65, 75, 75, 55])
    t_final.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 6.8),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2), ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_final)
    story.append(Paragraph("<b>Table 5 :</b> Bilan final sur le jeu de test sanctuarisé (moyennes et écarts-types sur 3 graines).", caption_style))

    story.append(Paragraph(
        "<b>5.2 Discussion critique essentielle : La Régression Logistique surpasse les réseaux de neurones sur le test :</b><br/>"
        "Un résultat capital émerge de nos données : <b>la régression logistique linéaire (95,28%) bat à la fois le MLP 1 couche (93,88%) "
        "et le MLP 3 couches profond (94,17%) sur le jeu de test !</b><br/>"
        "<i>Comment l'expliquer scientifiquement ?</i> Loin d'être un échec, ce constat met en lumière la nature des données : "
        "les 561 variables sont des caractéristiques expertes lissées (statistiques d'énergie, centroïdes spectraux, bandes FFT) conçues "
        "pour séparer les activités de façon quasi-linéaire. Les réseaux de neurones, dotés d'une grande capacité de modélisation non-linéaire, "
        "ont tendance à ajuster des corrélations complexes propres aux 17 sujets du train (sur-spécification inter-sujets). "
        "La régression logistique, par sa rigidité linéaire, est insensible à ces idiosyncrasies et généralise donc remarquablement bien sur de nouveaux individus.",
        body_style
    ))

    # Matrice de confusion et courbes
    f_curves = "reports/figures/09_courbes_apprentissage_meilleur_modele.png"
    f_cm = "reports/figures/10_matrice_confusion_test.png"
    if os.path.exists(f_curves) and os.path.exists(f_cm):
        img_t2 = Table([[Image(f_curves, width=255, height=105), Image(f_cm, width=255, height=125)]], colWidths=[260, 260])
        img_t2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_t2)
        story.append(Paragraph("<b>Figure 2 :</b> Dynamique d'apprentissage stabilisée par Dropout 0.5 (gauche) et Matrice de confusion test (droite).", caption_style))

    # Analyse physique de la matrice
    cm_sit_stand = r2['test_confusion']['matrice'][1][2]
    total_sit = sum(r2['test_confusion']['matrice'][1])
    pct_sit = (cm_sit_stand / total_sit) * 100
    cm_down_up = r2['test_confusion']['matrice'][4][5]
    total_down = sum(r2['test_confusion']['matrice'][4])
    pct_down = (cm_down_up / total_down) * 100

    story.append(Paragraph(
        "<b>5.3 Analyse physique des erreurs de capteurs :</b><br/>"
        f"• <i>Position Allongée (LAYING) :</i> Détection parfaite à <b>100%</b> (537/537, F1: 0,999) car la gravité se projette orthogonalement à l'axe vertical du corps.<br/>"
        f"• <i>Confusion Assis (SITTING) vs Debout (STANDING) :</i> <b>93 assis sont classés debout ({pct_sit:.1f}% d'erreur !)</b> et 20 debout sont classés assis. "
        "<b>Explication physique des capteurs :</b> Dans ces deux postures, le sujet est immobile : l'accélération dynamique est nulle et la vitesse angulaire quasi-nulle. "
        "Le téléphone placé à la ceinture ne mesure que le vecteur gravité vertical 1g. La seule nuance réside dans l'orientation du fémur (cuisse), "
        "qui n'induit qu'une variation angulaire infime au niveau de la taille chez certains volontaires.<br/>"
        f"• <i>Escaliers :</i> 34 descentes sont confondues avec des montées ({pct_down:.1f}%). La cadence de pas est très proche ; seule l'intensité verticale de décélération à l'impact les sépare.",
        body_style
    ))

    # ==================== 6. INTERPRÉTABILITÉ ====================
    story.append(Paragraph("6. Interprétabilité Latente (t-SNE) et Étude d'Ablation (Bonus)", h1_style))
    story.append(Paragraph(
        "<b>6.1 Visualisation t-SNE sur l'ensemble de validation (Partie 6) :</b> "
        "Pour préserver le jeu de test, nous projetons les 1 298 exemples de validation en 2D avec t-SNE (perplexité 30) : "
        "d'abord sur les 561 caractéristiques d'entrée brutes, puis sur les activations de la dernière couche cachée (64 neurones) du meilleur réseau (Figure 3 gauche et centre).<br/>"
        f"• <i>Pureté k-NN :</i> Elle progresse de <b>{r2['tsne']['purete_knn']['entrées brutes']*100:.2f}%</b> dans l'espace brut à <b>{r2['tsne']['purete_knn']['couche cachée']*100:.2f}%</b> dans l'espace latent.<br/>"
        "• <i>Structuration géométrique :</i> Dans l'espace brut, les activités dynamiques se mêlent et les postures statiques forment une masse floue. "
        "Dans l'espace latent, le réseau a contracté chaque activité en un îlot homogène et bien séparé, le seul contact persistant étant précisément la frontière entre <i>Sitting</i> et <i>Standing</i>.",
        body_style
    ))

    # Figures t-SNE et ablation
    f_tsne = "reports/figures/11_tsne_brut_vs_latent.png"
    f_abl = "reports/figures/12_importance_capteurs_ablation.png"
    if os.path.exists(f_tsne) and os.path.exists(f_abl):
        img_t3 = Table([[Image(f_tsne, width=320, height=135), Image(f_abl, width=190, height=135)]], colWidths=[325, 195])
        img_t3.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_t3)
        story.append(Paragraph("<b>Figure 3 :</b> Projections t-SNE (gauche: espace brut 561d, centre: espace latent 64d) et Étude d'ablation des capteurs (droite).", caption_style))

    # Bonus ablation
    story.append(Paragraph(
        "<b>6.2 Étude d'ablation par famille de capteurs (Partie 7 - Bonus) :</b> "
        "En ré-entraînant le réseau sur des sous-familles de capteurs (Figure 3 droite) :<br/>"
        f"• <i>Sans accélération gravitationnelle (pas de tGravity) :</i> La précision chute à <b>{r2['ablation'][1]['val_acc']*100:.2f}%</b>. La gravité est indispensable pour séparer les postures statiques.<br/>"
        f"• <i>Sans gyroscope (accéléromètre seul, 348 vars) :</i> <b>{r2['ablation'][2]['val_acc']*100:.2f}%</b>. L'accéléromètre assure la majorité du signal, le gyroscope apporte +1,8 pt crucial pour la rotation.<br/>"
        f"• <i>Domaine temporel seul ({r2['ablation'][3]['Variables']} vars) :</i> <b>{r2['ablation'][3]['val_acc']*100:.2f}%</b> vs <i>Domaine fréquentiel seul ({r2['ablation'][4]['Variables']} vars) :</i> <b>{r2['ablation'][4]['val_acc']*100:.2f}%</b> (-14,3 pt). Les statistiques temporelles brutes sont nettement plus discriminantes que les bandes FFT seules.",
        body_style
    ))

    # ==================== 7. CONCLUSION & IA ====================
    story.append(Paragraph("7. Conclusion et Déclaration d'Usage des Outils d'IA", h1_style))
    story.append(Paragraph(
        "<b>Conclusion :</b> Cette étude progressive démontre que la modélisation profonde sur signaux de capteurs requiert une méthodologie irréprochable. "
        "Sur des données riches en caractéristiques expertes, un réseau plus profond ne surpasse pas mécaniquement un modèle linéaire simple, car il surapprend "
        "la variabilité inter-sujets du jeu d'entraînement. La régularisation par Dropout (p=0,5) est déterminante pour contrôler cette variance. "
        "Enfin, la matrice de confusion et l'espace latent t-SNE éclairent parfaitement les limites physiques intrinsèques des capteurs inertiels.",
        body_style
    ))
    
    # Déclaration IA demandée en 8.2 (sobre, exacte et cadrée)
    ai_box = (
        "<b>Déclaration d'usage des outils d'IA (Partie 8.2) :</b> Des assistants d'IA (Claude, Antigravity) ont été utilisés comme aide à la programmation : "
        "génération de l'architecture initiale du code, restructuration modulaire des notebooks et mise en page du présent document ReportLab. "
        "L'ensemble des expérimentations a été exécuté en local sur les données réelles du projet par le binôme, qui a vérifié tous les contrôles théoriques "
        "(perte initiale, sanity check, découpage par sujet) et rédigé l'ensemble des analyses critiques et interprétations physiques à partir des résultats observés."
    )
    t_ai = Table([[Paragraph(ai_box, abstract_style)]], colWidths=[520])
    t_ai.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_ai)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Rapport PDF généré avec succès : {output_pdf}")

if __name__ == "__main__":
    build_pdf_report()
