import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Générateur de canvas personnalisé pour afficher le numéro de page 'Page X sur Y' et l'en-tête"""
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
        
        # En-tête (à partir de la page 2)
        if self._pageNumber > 1:
            self.drawString(40, 810, "M2 MALIA Deep Learning — Projet HAR : Étude progressive des réseaux de neurones")
            self.drawRightString(555, 810, "A. DIOP & M. FALL")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(40, 804, 555, 804)
            
        # Pied de page (sur toutes les pages)
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        self.drawString(40, 24, "Université Claude Bernard Lyon 1 — Année 2026–2027")
        self.drawRightString(555, 24, f"Page {self._pageNumber} sur {page_count}")
        self.restoreState()

def build_pdf_report(output_pdf="reports/M2_DL_2026_DIOP_FALL.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Centré
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=1,
        spaceAfter=6
    )
    
    author_style = ParagraphStyle(
        'DocAuthors',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=8
    )
    
    abstract_style = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
        alignment=4 # Justifié
    )
    
    h1_style = ParagraphStyle(
        'SecH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SecH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11.5,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=10.8,
        textColor=colors.HexColor("#2D3748"),
        alignment=4,
        spaceAfter=4
    )
    
    caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=2,
        spaceAfter=4
    )

    story = []

    # =========================================================================
    # EN-TÊTE DU PAPIER SCIENTIFIQUE
    # =========================================================================
    story.append(Paragraph("Étude Progressive des Réseaux de Neurones Multicouches (MLP) pour la Reconnaissance d'Activités Humaines (HAR)", title_style))
    story.append(Paragraph("Analyse Architecturale, Régularisation, Évaluation Rigoureuse et Interprétabilité Latente", subtitle_style))
    story.append(Paragraph("<b>Alioune DIOP & Madjiguène FALL</b><br/>Master 2 MALIA — Enseignant : Walid Bechkit — Année universitaire 2026–2027", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=5))

    # Résumé / Abstract dans un encadré
    abstract_text = (
        "<b>Résumé —</b> Ce travail présente une étude systématique de réseaux de neurones multicouches (MLP) pour la classification d'activités humaines "
        "à partir des capteurs inertiels de smartphones (HAR Dataset, 6 activités, 561 variables). En adoptant un protocole rigoureux sans fuite d'information "
        "(découpage par volontaires entiers plutôt qu'aléatoire), nous établissons d'abord une référence logistique (95,99% val / 95,28% test). "
        "Nous analysons ensuite pas à pas l'impact des fonctions d'activation (Tanh surpassant ReLU à 95,53%), des optimiseurs (SGD Momentum et Adam) "
        "et de la variabilité multi-graines. L'extension en profondeur (jusqu'à 4 couches) révèle un risque accru de surapprentissage, efficacement jugulé "
        "par la combinaison conjointe de Batch Normalization, Dropout (p=0,3) et Weight Decay (L2=1e-4), portant la précision finale sur le jeu de test sanctuarisé "
        "à plus de 96,5%. L'analyse d'interprétabilité par t-SNE démontre comment le réseau structure un espace latent hautement discriminant, séparant parfaitement "
        "les activités dynamiques et restreignant les légères confusions aux seules activités statiques posturales (Assis vs Debout)."
    )
    abs_table = Table([[Paragraph(abstract_text, abstract_style)]], colWidths=[520])
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(abs_table)
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 1 : INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. Introduction et Problématique", h1_style))
    story.append(Paragraph(
        "La reconnaissance d'activités humaines (<i>Human Activity Recognition</i> - HAR) à partir de capteurs inertiels miniatures (accéléromètres et gyroscopes) "
        "est au cœur de nombreuses applications sociétales majeures : télémédecine, assistance aux personnes âgées, analyse biomécanique et interfaces homme-machine contextuelles. "
        "Toutefois, la variabilité inter-individuelle des mouvements et la corrélation temporelle inhérente aux séries temporelles posent des défis méthodologiques fondamentaux.",
        body_style
    ))
    story.append(Paragraph(
        "Ce projet a pour ambition d'analyser pas à pas la construction de réseaux de neurones multicouches (MLP) sur des données réelles de capteurs. "
        "Plutôt que de chercher aveuglément la meilleure performance brute, notre démarche scientifique s'attache à comprendre et justifier l'influence de chaque composante "
        "(largeur, profondeur, activations, optimiseurs, régularisations, variabilité statistique). Pour ce faire, nous appliquons une méthodologie stricte : "
        "une référence simple (baseline), un ensemble de validation extrait sans fuite de données, des répétitions sur graines aléatoires multiples, et un ensemble de test réservé pour une évaluation finale unique.",
        body_style
    ))

    # =========================================================================
    # SECTION 2 : JEU DE DONNÉES & MÉTHODOLOGIE
    # =========================================================================
    story.append(Paragraph("2. Jeu de Données et Méthodologie Expérimentale", h1_style))
    story.append(Paragraph(
        "<b>2.1 Présentation des données :</b> Le jeu de données HAR (Anguita et al., UCI) est issu de 30 volontaires équipés d'un smartphone Samsung Galaxy S II à la taille. "
        "Les signaux tri-axiaux d'accélération et de vitesse angulaire ont été filtrés et découpés en fenêtres glissantes de 2,56 secondes (128 lectures à 50 Hz avec 50% de chevauchement). "
        "À partir de chaque fenêtre, <b>561 caractéristiques</b> ont été calculées dans les domaines temporel et fréquentiel (moyennes, écarts-types, bandes d'énergie, entropie, etc.). "
        "La tâche consiste à prédire l'une des <b>6 activités</b> suivantes : <i>WALKING</i>, <i>WALKING_UPSTAIRS</i>, <i>WALKING_DOWNSTAIRS</i>, <i>SITTING</i>, <i>STANDING</i> et <i>LAYING</i>.",
        body_style
    ))
    story.append(Paragraph(
        "Le fichier d'entraînement (<code>train.csv</code>) comprend <b>7 352 observations</b> réparties sur 21 volontaires, tandis que le fichier de test (<code>test.csv</code>) "
        "contient <b>2 947 observations</b> issues de 9 volontaires distincts. <b>Aucun volontaire n'est partagé</b> entre le train et le test, imposant ainsi d'évaluer la généralisation sur des personnes inconnues.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Prétraitements et questions de conception :</b><br/>"
        "• <i>Mise à l'échelle :</i> L'inspection des 561 variables révèle un minimum global de <b>-1,0000</b> et un maximum de <b>+1,0000</b> (moyenne globale: -0,508 ; écart-type moyen: 0,284). "
        "Toutes les caractéristiques ont déjà été normalisées dans [-1, 1] par les auteurs. Aucun scaling supplémentaire n'est requis.<br/>"
        "• <i>Colonnes d'entrée :</i> La colonne <code>subject</code> représente l'identifiant du volontaire. <b>Elle ne doit en aucun cas servir d'entrée au modèle</b>. "
        "L'inclure provoquerait une fuite de données majeure (*data leakage*) où le modèle mémoriserait les individus plutôt que les motifs de marche, ruinant sa généralisation sur de nouvelles personnes.<br/>"
        "• <i>Encodage de la cible :</i> Les 6 activités textuelles sont converties en indices entiers [0..5] via <code>LabelEncoder</code> pour alimenter la fonction de perte <code>nn.CrossEntropyLoss</code>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.3 Protocole de découpage : Aléatoire vs Par Volontaire :</b> "
        "Dans la littérature et les TP, un piège fréquent consiste à découper les lignes au hasard. Or, deux fenêtres temporelles consécutives d'un même individu se chevauchent à 50% et sont quasi-identiques. "
        "Pour le prouver, nous avons entraîné le modèle de référence selon deux découpages et comparé la validation au test réel :",
        body_style
    ))

    # Tableau découpage
    split_data = [
        ["Méthodologie de Découpage", "Validation Acc (%)", "Test Acc (%)", "Écart |Val - Test|", "Diagnostic"],
        ["Découpage Aléatoire (80/20)", "98,57 %", "96,13 %", "+2,44 %", "Estimation trop optimiste (Fuite de données)"],
        ["Découpage par Volontaires (Sujets 1, 3, 5, 7)", "95,99 %", "95,28 %", "0,71 %", "Fidèle à la réalité de généralisation"]
    ]
    t_split = Table(split_data, colWidths=[150, 85, 75, 75, 135])
    t_split.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (1,0), (3,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_split)
    story.append(Paragraph("<b>Table 1 :</b> Comparaison des stratégies de découpage sur la régression logistique de référence.", caption_style))

    # =========================================================================
    # SECTION 3 : PREMIÈRE ARCHITECTURE MLP & VARIABILITÉ
    # =========================================================================
    story.append(Paragraph("3. Première Architecture MLP et Résultats de Base (Version 1)", h1_style))
    story.append(Paragraph(
        "<b>3.1 Modèle de référence et contrôles préalables :</b> "
        "Avant d'explorer les réseaux profonds, nous établissons la régression logistique comme référence simple (Table 1, Val Acc: 95,99%). "
        "Nous concevons ensuite un MLP à <b>une seule couche cachée</b> (561 entrées, 64 neurones cachés, 6 sorties). Avant tout entraînement, deux vérifications rigoureuses sont opérées :<br/>"
        "1. <i>Perte initiale théorique :</i> Pour un réseau aléatoire sur C=6 classes équiprobables, la probabilité initiale p ≈ 1/6 donne une perte théorique L_0 = -ln(1/6) = ln(6) ≈ 1,7918. "
        "La valeur mesurée empiriquement sur le premier batch est de <b>1,7592</b>, validant la cohérence de l'initialisation.<br/>"
        "2. <i>Sanity check (surapprentissage sur mini-lot) :</i> Sur un sous-ensemble de 10 exemples, le réseau atteint <b>100,0%</b> de précision et une perte résiduelle de 0,0029 en 40 itérations, prouvant le bon calcul des gradients.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3.2 Étude des fonctions d'activation et algorithmes d'optimisation :</b> "
        "Nous comparons ReLU, Tanh et Sigmoid couplées à SGD, SGD avec Momentum (0,9) et Adam :",
        body_style
    ))

    # Tableau activations et optimiseurs
    res_v1_data = [
        ["Activation", "Optimiseur", "Learning Rate", "Paramètres", "Val Loss", "Val Accuracy", "Val Macro-F1"],
        ["ReLU", "Adam", "0.001", "36 358", "0.1346", "94,84 %", "94,61 %"],
        ["Tanh", "Adam", "0.001", "36 358", "0.1333", "95,38 %", "95,19 %"],
        ["Sigmoid", "Adam", "0.001", "36 358", "0.1386", "94,45 %", "94,22 %"],
        ["ReLU", "SGD standard", "0.05", "36 358", "0.2107", "91,45 %", "91,12 %"],
        ["ReLU", "SGD + Momentum", "0.01", "36 358", "0.1246", "95,53 %", "95,34 %"]
    ]
    t_v1 = Table(res_v1_data, colWidths=[65, 85, 65, 60, 55, 65, 65])
    t_v1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_v1)
    story.append(Paragraph("<b>Table 2 :</b> Performances comparatives des fonctions d'activation et optimiseurs (MLP 1 couche cachée, 64 neurones).", caption_style))

    story.append(Paragraph(
        "<b>3.3 Variabilité statistique multi-graines :</b> Les scores étant très resserrés (>94%), évaluer un modèle sur une seule graine expose au risque d'une fluctuation aléatoire. "
        "En répétant l'entraînement sur 3 graines (seeds 42, 123, 999) :<br/>"
        "• <i>ReLU + Adam :</i> 94,97 % ± 0,18 % (scores : 94,84%, 94,84%, 95,22%)<br/>"
        "• <i>Tanh + Adam :</i> 95,53 % ± 0,11 % (scores : 95,38%, 95,61%, 95,61%)<br/>"
        "L'écart moyen (+0,56%) dépasse les écarts-types (0,18% et 0,11%). L'avantage de <b>Tanh</b> est statistiquement avéré sur cette architecture simple, "
        "car les variables d'entrée étant normalisées dans [-1, 1], la symétrie de Tanh autour de zéro facilite la transmission des gradients.",
        body_style
    ))

    # =========================================================================
    # SECTION 4 : EXTENSION DE L'ARCHITECTURE & RÉGULARISATION
    # =========================================================================
    story.append(Paragraph("4. Extension de l'Architecture et Régularisation (Version 2)", h1_style))
    story.append(Paragraph(
        "<b>4.1 Impact de la largeur et de la profondeur :</b> "
        "Nous explorons l'effet de l'augmentation du nombre de neurones par couche (128, 256) et du nombre de couches cachées (1 à 4 couches) :",
        body_style
    ))

    arch_table_data = [
        ["Architecture", "Couches Cachées", "Paramètres", "Temps (s)", "Train Loss", "Val Loss", "Val Acc (%)", "Val F1 (%)"],
        ["MLP 1 couche (64) [V1]", "64", "36 358", "12.8s", "0.0812", "0.1346", "94,84 %", "94,61 %"],
        ["MLP 1 couche (128)", "128", "72 710", "13.4s", "0.0451", "0.1287", "95,45 %", "95,26 %"],
        ["MLP 1 couche (256)", "256", "145 414", "15.1s", "0.0210", "0.1310", "95,61 %", "95,40 %"],
        ["MLP 2 couches [128, 64]", "128 -> 64", "80 966", "14.2s", "0.0234", "0.1295", "95,76 %", "95,58 %"],
        ["MLP 3 couches [256, 128, 64]", "256 -> 128 -> 64", "185 414", "18.5s", "0.0104", "0.1412", "95,84 %", "95,62 %"],
        ["MLP 4 couches [256, 128, 64, 32]", "256 -> 128 -> 64 -> 32", "187 494", "21.2s", "0.0082", "0.1580", "95,22 %", "95,01 %"]
    ]
    t_arch = Table(arch_table_data, colWidths=[120, 85, 55, 45, 50, 45, 55, 55])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.0),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_arch)
    story.append(Paragraph("<b>Table 3 :</b> Évolution des métriques en fonction de la complexité architecturale (sans régularisation explicite).", caption_style))

    story.append(Paragraph(
        "<i>Constat sur l'overfitting :</i> À mesure que la profondeur s'accroît, la perte d'entraînement s'effondre (0,0082 pour 4 couches), "
        "tandis que la perte de validation remonte (0,1580). Le modèle à 4 couches commence à surapprendre les particularités des 17 volontaires d'entraînement. "
        "Ce phénomène justifie pleinement la mise en œuvre de régularisations puissantes.",
        body_style
    ))

    story.append(Paragraph(
        "<b>4.2 Évaluation des techniques de régularisation :</b> "
        "Sur l'architecture 3 couches (256-128-64), nous évaluons le Dropout (p=0,1, 0,3, 0,5), le Weight Decay (L2 = 1e-4), la Batch Normalization (BN) et l'Early Stopping :",
        body_style
    ))

    reg_table_data = [
        ["Configuration de Régularisation", "Val Loss", "Val Accuracy", "Val Macro-F1", "Effet Principal Observé"],
        ["Sans régularisation (Baseline 3c)", "0.1412", "95,84 %", "95,62 %", "Surapprentissage rapide (train loss quasi-nulle)"],
        ["Dropout (p = 0.1)", "0.1290", "96,15 %", "95,98 %", "Réduction modérée de la variance"],
        ["Dropout (p = 0.3)", "0.1185", "96,61 %", "96,44 %", "Excellent compromis biais/variance"],
        ["Dropout (p = 0.5)", "0.1342", "95,92 %", "95,73 %", "Sous-apprentissage partiel (trop d'extinctions)"],
        ["Weight Decay (L2 = 1e-4)", "0.1255", "96,22 %", "96,05 %", "Poids plus lisses, évite les extrêmes"],
        ["Batch Normalization (BN)", "0.1150", "96,76 %", "96,58 %", "Convergence accélérée, gradient stable"],
        ["<b>BatchNorm + Dropout 0.3 + L2 (1e-4)</b>", "<b>0.1042</b>", "<b>97,15 %</b>", "<b>96,98 %</b>", "<b>Meilleure régularisation globale, haute stabilité</b>"]
    ]
    t_reg = Table(reg_table_data, colWidths=[150, 50, 60, 60, 190])
    t_reg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('ALIGN', (1,0), (3,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_reg)
    story.append(Paragraph("<b>Table 4 :</b> Comparaison des mécanismes de régularisation sur le MLP à 3 couches cachées.", caption_style))

    # =========================================================================
    # SECTION 5 : ÉVALUATION FINALE SUR TEST & MATRICE DE CONFUSION
    # =========================================================================
    story.append(Paragraph("5. Visualisation des Courbes et Évaluation Finale sur Test", h1_style))
    story.append(Paragraph(
        "<b>5.1 Courbes d'apprentissage :</b> La Figure 1 illustre la dynamique d'entraînement du meilleur modèle régularisé (BatchNorm + Dropout 0.3 + L2). "
        "Contrairement au modèle non régularisé où la perte de validation divergeait dès l'époque 10, les courbes d'entraînement et de validation progressent de façon conjointe et stable. "
        "L'Early Stopping stabilise le modèle à son optimum sans dégradation.",
        body_style
    ))

    # Insertion des figures 09 et 10 si disponibles
    f09 = "reports/figures/09_courbes_apprentissage_meilleur_modele.png"
    f10 = "reports/figures/10_matrice_confusion_test.png"
    if os.path.exists(f09) and os.path.exists(f10):
        img_table = Table([[
            Image(f09, width=255, height=110),
            Image(f10, width=255, height=130)
        ]], colWidths=[260, 260])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(img_table)
        story.append(Paragraph("<b>Figure 1 :</b> Courbes d'apprentissage (gauche) et Matrice de confusion sur le jeu de test final (droite).", caption_style))

    story.append(Paragraph(
        "<b>5.2 Évaluation finale (Passage unique sur test.csv) :</b> "
        "En respectant strictement la consigne de n'évaluer le jeu de test qu'une seule fois à l'issue de tous les choix d'hyperparamètres, nous comparons nos modèles retenus :",
        body_style
    ))

    test_summary_data = [
        ["Modèle Évalué", "Val Acc (%)", "Test Acc (%)", "Test F1 (%)", "Écart |Val - Test|", "Statut / Rôle"],
        ["1. Régression Logistique", "95,99 %", "95,28 %", "95,12 %", "0,71 %", "Référence simple (Partie 1)"],
        ["2. MLP 1 couche [64] (Tanh)", "95,53 %", "95,15 %", "94,98 %", "0,38 %", "Baseline réseau (Partie 2)"],
        ["3. Deep MLP 3 couches Régularisé", "97,15 %", "<b>96,54 %</b>", "<b>96,41 %</b>", "<b>0,61 %</b>", "<b>Architecture finale retenue</b>"]
    ]
    t_test = Table(test_summary_data, colWidths=[150, 75, 75, 75, 75, 60])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('ALIGN', (1,0), (4,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_test)
    story.append(Paragraph("<b>Table 5 :</b> Bilan final comparatif des architectures sur l'ensemble de test sanctuarisé.", caption_style))

    story.append(Paragraph(
        "<b>5.3 Analyse physique de la matrice de confusion :</b><br/>"
        "L'examen détaillé de la matrice de confusion (Figure 1 droite) fournit des enseignements physiques cruciaux :<br/>"
        "• <i>Activités dynamiques (Marche, Montée, Descente) :</i> Prédites avec une quasi-perfection (>98%). Les motifs périodiques et l'énergie spectrale du gyroscope discriminent nettement ces mouvements.<br/>"
        "• <i>Position Allongée (LAYING) :</i> Prédite à 100%. L'orientation de l'accélération gravitationnelle (composante de pesanteur projetée sur l'axe X ou Y du téléphone plutôt que Z) est totalement univoque.<br/>"
        "• <i>Confusion Assis (SITTING) vs Debout (STANDING) :</i> Constitue la quasi-totalité des erreurs résiduelles (environ 3 à 4% d'ambiguïté mutuelle). "
        "<b>Explication physique des capteurs :</b> Dans les deux cas, le sujet est immobile (accélération dynamique proche de zéro). Le smartphone, placé à la hanche ou dans la poche, "
        "mesure un vecteur gravité vertical très similaire dans les deux postures. Seule une légère inclinaison de la cuisse permet de les distinguer, ce qui explique la difficulté même pour un réseau profond.",
        body_style
    ))

    # =========================================================================
    # SECTION 6 : INTERPRÉTABILITÉ & RÉDUCTION DE DIMENSION
    # =========================================================================
    story.append(Paragraph("6. Interprétabilité et Réduction de Dimension (t-SNE)", h1_style))
    story.append(Paragraph(
        "<b>6.1 Principe et précautions d'interprétation :</b> "
        "La méthode t-SNE (<i>t-Distributed Stochastic Neighbor Embedding</i>) projette des données de haute dimension dans un espace 2D en préservant les voisinages locaux "
        "par minimisation de la divergence de Kullback-Leibler. Si t-SNE excelle à révéler des clusters, deux précautions s'imposent : "
        "(1) les distances relatives entre clusters éloignés n'ont pas de signification métrique absolue ; (2) la topologie dépend de la perplexité (fixée ici à 35).",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.2 Espace d'entrée brut vs Espace latent appris :</b> "
        "Nous appliquons t-SNE sur un sous-ensemble de 1 200 exemples de test, d'abord sur les 561 caractéristiques brutes, puis sur les activations de la dernière couche cachée (64 neurones) du réseau :",
        body_style
    ))

    f11 = "reports/figures/11_tsne_brut_vs_latent.png"
    if os.path.exists(f11):
        story.append(Image(f11, width=515, height=185))
        story.append(Paragraph("<b>Figure 2 :</b> Projections t-SNE des données de test : variables d'entrée brutes (gauche) vs espace latent appris (droite).", caption_style))

    story.append(Paragraph(
        "<b>6.3 Analyse comparative :</b><br/>"
        "• <i>Dans l'espace brut (Figure 2 gauche) :</i> Si les activités dynamiques (marche en nuances de vert/bleu) forment un groupe distinct des activités statiques, "
        "les classes <i>SITTING</i> et <i>STANDING</i> se chevauchent de manière très confuse, et les montées/descentes d'escaliers s'entremêlent.<br/>"
        "• <i>Dans l'espace latent appris (Figure 2 droite) :</i> Le réseau a profondément réorganisé la représentation. Les 6 activités apparaissent sous forme d'îlots compacts et nettement séparés. "
        "Le seul point de tangence subsistant se situe précisément entre les clusters <i>SITTING</i> et <i>STANDING</i>, confirmant visuellement ce que nous avions quantifié sur la matrice de confusion. "
        "Le réseau a donc bien appris une transformation géométrique contractante sur chaque classe et dilatante entre classes distinctes.",
        body_style
    ))

    # =========================================================================
    # SECTION 7 : BONUS & CONCLUSION
    # =========================================================================
    story.append(Paragraph("7. Ouvertures (Bonus) et Conclusion Générale", h1_style))
    story.append(Paragraph(
        "<b>7.1 Bonus — Étude d'ablation par famille de capteurs :</b> "
        "Pour aller plus loin, nous avons mené une étude d'ablation en ré-entraînant le modèle sur des sous-ensembles spécifiques de capteurs :<br/>"
        "• <i>Tous les capteurs (561 vars) :</i> 95,84% val.<br/>"
        "• <i>Sans accélération de pesanteur (pas de tGravity) :</i> Chute spectaculaire à <b>83,4%</b> ! Sans la gravité, impossible de séparer SITTING, STANDING et LAYING.<br/>"
        "• <i>Sans gyroscope (accéléromètre seul, 345 vars) :</i> <b>93,12%</b>. L'accéléromètre porte l'essentiel de l'information, mais le gyroscope apporte les +2,7% décisifs pour la dynamique de rotation.<br/>"
        "• <i>Domaine fréquentiel seul (Fourier seul, 289 vars) :</i> <b>91,85%</b> vs <b>94,20%</b> pour le domaine temporel seul.",
        body_style
    ))

    f12 = "reports/figures/12_importance_capteurs_ablation.png"
    if os.path.exists(f12):
        story.append(Image(f12, width=420, height=130))
        story.append(Paragraph("<b>Figure 3 :</b> Étude d'ablation montrant le rôle prépondérant de la composante gravitationnelle.", caption_style))

    story.append(Paragraph(
        "<b>7.2 Conclusion générale :</b> "
        "Cette étude démontre que la réussite d'un modèle d'apprentissage profond ne repose pas uniquement sur l'augmentation du nombre de couches, "
        "mais avant tout sur une <b>méthodologie expérimentale rigoureuse</b> :<br/>"
        "1. Le découpage par individu est impératif pour éviter une surestimation illusoire liée à l'autocorrélation temporelle.<br/>"
        "2. Les réseaux profonds sont sensibles au surapprentissage sur des données tabulaires/capteurs, rendant la régularisation conjointe (BatchNorm + Dropout + L2) indispensable (+1,3% de gain net).<br/>"
        "3. L'analyse de l'espace latent (t-SNE) et l'analyse physique des capteurs corroborent parfaitement la matrice de confusion, expliquant pourquoi la posture statique demeure l'unique défi résiduel.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Mention d'usage des outils d'assistance :</b> "
        "Conformément aux directives de la Partie 8.2, l'assistance IA (Antigravity DeepMind) a été utilisée pour l'automatisation de la mise en page ReportLab et la vectorisation des graphiques. "
        "L'ensemble du code, des pipelines PyTorch, des expérimentations et des analyses critiques a été vérifié et validé par le binôme.",
        caption_style
    ))

    # Compilation
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Rapport PDF généré avec succès : {output_pdf}")

if __name__ == "__main__":
    build_pdf_report()
