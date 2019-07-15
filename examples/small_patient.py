"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>
	  
  File:     ep-pyg
  Authors:  Michio Honda (micchie.gml@gmail.com)

NEC Laboratories Europe GmbH, Copyright (c) 2020, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
import networkx as nx
import pandas as pd
import nltk
from typing import Collection, Mapping, Optional, List, Sequence, Tuple
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, EP

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x
#
# from load_small_patient_graph
#
ENGLISH_SNOWBALL_STEMMER = nltk.stem.snowball.SnowballStemmer("english")
"""English snowball stemmer from `nltk`"""
STRING_PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)
"""A mapping from 'punctuation' characters to `None`"""

ENGLISH_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
"""English stopwords from `nltk`"""

def clean_doc(
        doc:str,
        replacement_table:Optional[Mapping]=None,
        stop_words:Optional[Collection]=None,
        stemmer:nltk.stem.StemmerI=ENGLISH_SNOWBALL_STEMMER) -> str:

    if replacement_table is None:
        replacement_table = STRING_PUNCTUATION_TABLE
        
    if stop_words is None:
        stop_words = ENGLISH_STOP_WORDS
        
    # tokenize into words
    words = nltk.word_tokenize(doc)

    # convert to lower case
    words = [w.lower() for w in words]

    # remove punctuation from each word
    words = [w.translate(replacement_table) for w in words]

    # remove non-alphabetic words
    words = [w for w in words if w.isalpha()]

    # filter stopwords
    words = [w for w in words if not w in stop_words]

    # stem
    words = [stemmer.stem(w) for w in words]
    
    # join back
    words = ' '.join(words)
    
    return words

def load_small_patient_graph(greps=False):
    """ Get the graph for the simple patient example
    
    Returns
    -------
    patient_graph: networkx.Graph
        The graph
    """
    
    patient_graph = nx.Graph()

    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    patient_graph.add_nodes_from(nodes)

    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('B', 'F'),
        ('C', 'D'),
        ('C', 'E'),
        ('F', 'G'),
        ('F', 'H'),
        ('G', 'H'),
        ('G', 'I')
    ]
    patient_graph.add_edges_from(edges)

    
    return patient_graph

def load_small_patient_data():
    """ Initialize the data frame for the simple patient example
    
    Returns
    -------    
    patient_df: pd.DataFrame
        The data frame
    """

    # import this here to keep the namespace clean
    from sklearn.feature_extraction.text import CountVectorizer

    # give the patients some values

    gender = {
        'A': 'Male',
        'B': 'Female',
        'C': 'Male',
        'D': 'Male',
        'E': 'Female',
        'F': 'Female',
        'G': 'Female',
        'H': 'Female',
        'I': 'Male'
    }

    race = {
        'A': 'White',
        'B': 'White',
        'C': 'Asian',
        'D': 'White',
        'E': 'Asian',
        'F': 'White',
        'G': 'Black',
        'H': 'Black',
        'I': 'Black'
    }

    blood_pressure = {
        'A': 81.0,
        'B': 90.0,
        'C': 83.0,
        'D': 88.0,
        'E': 75.0,
        'F': 95.0,
        'G': 85.0,
        'H': 97.0,
        'I': 83.0
    }

    heart_rate = {
        'A': 93.0,
        'B': 75.0,
        'C': 67.0,
        'D': 80.0,
        'E': 63.0,
        'F': 99.0,
        'G': 76.0,
        'H': 65.0,
        'I': 75.0
    }
    
    weight = {
        'A': 40.5,
        'B': 75.3,
        'C': 67.6,
        'D': 80.1,
        'E': 63.3,
        'F': 30.6,
        'G': 76.5,
        'H': 65.2,
        'I': 75.1
    }
    
    #Free text
    phrases = {
        'A': 'One step towards success',
        'B': 'This is a good one',
        'C': 'Hello World',
        'D': 'Have Fun',
        'E': 'Life is unpredicable',
        'F': 'World is wonderful',
        'G': 'Enjoy every moment of life',
        'H': 'Health is wealth',
        'I': 'Testing EP package'
    }

    # cm: cardiomyopathy
    has_cm = {
        'A': True,
        'B': False,
        'C': False,
        'D': False,
        'E': False,
        'F': True,
        'G': True,
        'H': False,
        'I': False
    }
    
    # create some notes. the text is originally based on 20newsgroups
    documents = {
        'A': """Subject: Re: Lung disorders and clubbing of fingers\nArticle-I.D.: pitt.19424\nReply-To: km@cs.pitt.edu (Ken Mitchum)\nOrganization: Univ. of Pittsburgh Computer Science\nLines: 36\n\nIn article <SLAGLE.93Mar26205915@sgi417.msd.lmsc.lockheed.com> slagle@lmsc.lockheed.com writes:\n>Can anyone out there enlighten me on the relationship between\n>lung disorders and "clubbing", or swelling and widening, of the\n>fingertips?  What is the mechanism and why would a physician\n>call for chest xrays to diagnose the cause of the clubbing?\n\nPurists often distinguish between "true" clubbing and "pseudo"\nclubbing, the difference being that with "true" clubbing the\nangle of the nail when viewed from the side is constantly\nnegative when proceeding distally (towards the fingertip).\nWith "pseudo" clubbing, the angle is initially positive, then\nnegative, which is the normal situation. "Real" internists\ncan talk for hours about clubbing. I\'m limited to a couple\nof minutes.\n\nWhether this distinction has anything to do with reality is\nentirely unclear, but it is one of those things that internists\nlove to paw over during rounds. Supposedly, only "true" clubbing\nis associated with disease. The problem is that the list of\ndiseases associated with clubbing is quite long, and includes\nboth congenital conditions and acquired disease. Since many of\nthese diseases are associated with cardiopulmonary problems\nleading to right to left shunts and chronic hypoxemia, it is\nvery reasonable to get a chest xray. However, many of the \ncongenital abnormalities would only be diagnosed with a cardiac\ncatheterization. \n\nThe cause of clubbing is unclear, but presumably relates to\nsome factor causing blood vessels in the distal fingertip to\ndilate abnormally. \n\nClubbing is one of those things from an examination which is\na tipoff to do more extensive examination. Often, however,\nthe cause of the clubbing is quite apparent.\n\n-km\n'""",
        'B': """Subject: Eco-Freaks forcing Space Mining.\nArticle-I.D.: aurora.1993Apr21.212202.1\nOrganization: University of Alaska Fairbanks\nLines: 24\nNntp-Posting-Host: acad3.alaska.edu\n\nHere is a way to get the commericial companies into space and mineral\nexploration.\n\nBasically get the eci-freaks to make it so hard to get the minerals on earth..\nYou think this is crazy. Well in a way it is, but in a way it is reality.\n\nThere is a billin the congress to do just that.. Basically to make it so\nexpensive to mine minerals in the US, unless you can by off the inspectors or\ntax collectors.. ascially what I understand from talking to a few miner friends \nof mine, that they (the congress) propose to have a tax on the gross income of\nthe mine, versus the adjusted income, also the state governments have there\nnormal taxes. So by the time you get done, paying for materials, workers, and\nother expenses you can owe more than what you made.\nBAsically if you make a 1000.00 and spend 500. ofor expenses, you can owe\n600.00 in federal taxes.. Bascially it is driving the miners off the land.. And\nthe only peopel who benefit are the eco-freaks.. \n\nBasically to get back to my beginning statement, is space is the way to go\ncause it might just get to expensive to mine on earth because of either the\neco-freaks or the protectionist.. \nSuch fun we have in these interesting times..\n\n==\nMichael Adams, nsmca@acad3.alaska.edu -- I'm not high, just jacked\n""", # female
        'C': """Subject: GETTING AIDS FROM ACUPUNCTURE NEEDLES\nOrganization: University of California, Berkeley\nLines: 44\nDistribution: world\nNNTP-Posting-Host: uclink.berkeley.edu\n\n   someone wrote in expressing concern about getting AIDS from acupuncture\n   needles.....\n\nUnless your friend is sharing fluids with their acupuncturist who               \nthemselves has AIDS..it is unlikely (not impossible) they will get AIDS         \nfrom acupuncture needles. Generally, even if accidently inoculated, the normal\nimmune response should be enough to effectively handle the minimal contaminant \ninvolved with acupuncture needle insertion. \n\nMost acupuncturists use disposable needles...use once and throw away. They      \ndo this because you are not the only one concerned about transmission of \ndiseases via this route...so it\'s good business to advertise "disposable needlesused here." These needles tend to be of a lower quality however, \nbeing poorly manufactured and too "sharp" in my opinion. They tend to snag bloodvessels on insertion compared to higher quality needles.                                                                        \nIf I choose to use acupuncture for a given complaint, that patient will get \ntheir own set of new needles which are sterilized between treatments.      \nThe risk here for hepatitis, HIV, etc. transmission is that I could mistakenly \nuse an infected persons needles accidently on the wrong              \npatient...but clear labelling and paying attention all but eliminates \nthis risk. Better quality needles tend to "slide" past vessels and            \nnerves avoiding unpleasant painful snags..and hematomas...so I use them.                        \nAcupuncture needles come in many lengths and thicknesses...but they are all \nsolid when compared to their injection-style cousins. In China, herbal solutionsand western pharmaceuticals are occasionally injected into \nmeridian points purported to have TCM physiologic effects and so require \nthe same hollow needles used for injecting fluid medicine. This means...thinkingtiny...that a samll amount of tissue, the diameter of the needle bore, will be \ninjected into the body as it would  be in a typical "shot." when the skin is \npuntured. On the other hand when the solid \nacupuncture needle is inserted, the skin tends to "squeeze" the needle \nfrom the tip to the level of insertion such that any \'cooties\' that \nhaven\'t been schmeared away with alcohol before insertion, tend to remain \non the surface of the skin minimizing invasion from the exterior. \n\nOf course in TCM...the body\'s exterior is protected by the Wei (Protective) Qi..so infection is unlikely....or in other words...there is a normal inflammatory \nand immune response that accompanies tissue damage incurred at the puncture \nsite.\n\n\nWhile I\'m fairly certain your friend will not have a transferable disease \ntransmitted to them via acupuncture needle insertion, I would like to know for \nwhat complaint they have consulted the acupuncturist...not to know  if it would be harmful.. but to know if it would be helpful. \n\nJohn Badanes, DC, CA\nromdas@uclink.berkeley.edu\n                                                                                                    \n  \n'""",# # m
        'D': """Subject: Re: What are the problems with Nutrasweet (Aspartame)?\nOrganization: The Portal System (TM)\nLines: 11\n\nPhenylketonuria is a disease in which the body cannot process phenylalanine.\nIt can build up in the blood and cause seizures and neurological damage.\nAn odd side effect is that the urine can be deeply colored, like red wine.\nPeople with the condition must avoid Nutrasweet, chocolate, and anything\nelse rich in phenylalanine.\n\nAspartame is accused of having caused various vague neurological symptoms.\nPat Robertson's program _The_700_Club_ was beating the drum against\naspartame rather vigorously for about a year, but that issue seems to\nhave been pushed to the back burner for the last year or so.  Apparently,\nthe evidence is not very strong, or Pat would still be flailing away.\n""",# # m
        'E': """Subject: Re: Sunrise/ sunset times\nOrganization: Express Access Online Communications USA\nLines: 18\nNNTP-Posting-Host: access.digex.net\n\nIn article <1r6f3a$2ai@news.umbc.edu> rouben@math9.math.umbc.edu (Rouben Rostamian) writes:\n>how the length of the daylight varies with the time of the year.\n>Experiment with various choices of latitudes and tilt angles.\n>Compare the behavior of the function at locations above and below\n>the arctic circle.\n\n\n\nIf you want to have some fun.\n\nPlug the basic formulas  into Lotus.\n\nUse the spreadsheet auto re-calc,  and graphing functions\nto produce  bar graphs  based on latitude,  tilt  and hours of day light avg.\n\n\npat\n\n""", # f
        'F': """Subject: Re: Proton/Centaur?\nOrganization: Motorola\nNntp-Posting-Host: 145.1.146.43\nLines: 37\n\nIn article <1993Apr20.211638.168730@zeus.calpoly.edu> jgreen@trumpet.calpoly.edu (James Thomas Green) writes:\n>Has anyone looked into the possiblity of a Proton/Centaur combo?\n>What would be the benefits and problems with such a combo (other\n>than the obvious instability in the XSSR now)?\n\nI haven't seen any speculation about it. But, the Salyut KB (Design Bureau) \nwas planning a new LH/LOX second stage for the Proton which would boost\npayload to LEO from about 21000 to 31500 kg. (Geostationary goes from\n2600 kg. (Gals launcher version) to 6000 kg.. This scheme was competing\nwith the Energia-M last year and I haven't heard which won, except now\nI recently read that the Central Specialized KB was working on the \nsuccessor to the Soyuz booster which must be the Energia-M. So the early\nresults are Energia-M won, but this is a guess, nothing is very clear in \nRussia. I'm sure if Salyut KB gets funds from someone they will continue \ntheir development. \n\nThe Centaur for the Altas is about 3 meters dia. and the Proton \nis 4 so that's a good fit for their existing upper stage, the Block-D\nwhich sets inside a shround just under 4 meters dia. I don't know about\nlaunch loads, etc.. but since the Centaur survives Titan launches which\nare probably worse than the Proton (those Titan SRB's probably shake things\nup pretty good) it seems feasible. EXCEPT, the Centaur is a very fragile\nthing and may require integration on the pad which is not available now.\nProtons are assembled and transported horizontially. Does anyone know \nhow much stress in the way of a payload a Centaur could support while\nbolted to a Proton horizontally and then taken down the rail road track\nand erected on the pad?  \n\nThey would also need LOX and LH facilities added to the Proton pads \n(unless the new Proton second stage is actually built), and of course\nany Centaur support systems and facilities, no doubt imported from the\nUS at great cost. These systems may viloate US law so there are political\nproblems to solve in addition to the instabilities in the CIS you mention. \n\nDennis Newkirk (dennisn@ecs.comm.mot.com)\nMotorola, Land Mobile Products Sector\nSchaumburg, IL\n""", # f
        'G': """Subject: Re: Life on Mars???\nOrganization: U of Toronto Zoology\nLines: 24\n\nIn article <1993Apr20.120311.1@pa881a.inland.com> schiewer@pa881a.inland.com (Don Schiewer) writes:\n>What is the deal with life on Mars?  I save the "face" and heard \n>associated theories. (which sound thin to me)\n\nThe "face" is an accident of light and shadow.  There are many "faces" in\nlandforms on Earth; none is artificial (well, excluding Mount Rushmore and\nthe like...).  There is also a smiley face on Mars, and a Kermit The Frog.\n\nThe question of life in a more mundane sense -- bacteria or the like -- is\nnot quite closed, although the odds are against it, and the most that the\nmore orthodox exobiologists are hoping for now is fossils.\n\nThere are currently no particular plans to do any further searches for life.\n\n>Are we going back to Mars to look at this face agian?\n\nMars Observer, currently approaching Mars, will probably try to get a better\nimage or two of the "face" at some point.  It\'s not high priority; nobody\ntakes it very seriously.  The shadowed half of the face does not look very\nface-like, so all it will take is one shot at a different sun angle to ruin\nthe illusion.\n-- \nAll work is one man\'s work.             | Henry Spencer @ U of Toronto Zoology\n                    - Kipling           |  henry@zoo.toronto.edu  utzoo!henry""", # f
        'H': """Subject: Re: space food sticks\nKeywords: food\nLines: 25\nNntp-Posting-Host: skndiv.dseg.ti.com\nReply-To: pyron@skndiv.dseg.ti.com\nOrganization: TI/DSEG VAX Support\n\n\nIn article <C50z77.EE6@news.cso.uiuc.edu>, jelson@rcnext.cso.uiuc.edu (John Elson) writes:\n>Has anyone ever heard of a food product called "Space Food Sticks?" This\n>was apparently created/marketed around the time of the lunar expeditions, along\n>with "Tang" and other dehydrated foods. I have spoken with several people\n>who have eaten these before, and they described them as a dehydrated candy. \n>Any information would be greatly appreciated. \n\nA freeze dried Tootsie Roll (tm).  The actual taste sensation was like nothing\nyou will ever willingly experience.  The amazing thing was that we ate a second\none, and a third and ....\n\nI doubt that they actually flew on missions, as I\'m certain they did "bad\nthings" to the gastrointestinal tract.  Compared to Space Food Sticks, Tang was\na gastronomic contribution to mankind.\n--\nDillon Pyron                      | The opinions expressed are those of the\nTI/DSEG Lewisville VAX Support    | sender unless otherwise stated.\n(214)462-3556 (when I\'m here)     |\n(214)492-4656 (when I\'m home)     |God gave us weather so we wouldn\'t complain\npyron@skndiv.dseg.ti.com          |about other things.\nPADI DM-54909                     |\n\nPS. I don\'t think Tang flew, either.  Although it was developed under contract.\n\n""", # f
        'I': """Subject: Re: Lactose intolerance\nOrganization: Immunex Corporation, Seattle, WA\nLines: 27\n\nIn article <1993Apr5.165716.59@immunex.com>, rousseaua@immunex.com writes:\n> In article <ng4.733990422@husc.harvard.edu>, ng4@husc11.harvard.edu (Ho Leung Ng) writes:\n>> \n>>    When I was a kid in primary school, I used to drink tons of milk without\n>> any problems.  However, nowadays, I can hardly drink any at all without\n>> experiencing some discomfort.  What could be responsible for the change?\n>> \n>> Ho Leung Ng\n>> ng4@husc.harvard.edu\n\n\nOOPS. My original message died. I'll try again...\nI always understood (perhaps wrongly...:)) that the bacteria in our digestive\ntracts help us break down the components of milk. Perhaps the normal flora of \nthe intestine changes as one passes from childhood.\nIs there a pathologist or microbiologist in the house?\n\nAnne-Marie Rousseau\ne-mail: rousseaua@immunex.com\n(Please note that these opinions are mine, and only mine.)\n\n         \n            \n           \n           \n\n\n""", # m
    }

    # trim, remove stopwords, etc.
    cleaned_docs = {
        k: clean_doc(v) for k,v in documents.items()
    }

    # format for the CountVectorizer
    train_docs = [
        cleaned_docs[k] for k in sorted(cleaned_docs.keys())
    ]

    # convert to a sparse token count matrix
    count_vectorizer = CountVectorizer(min_df=0.2)
    count_vectorizer_fit = count_vectorizer.fit(train_docs)
    document_tokens = count_vectorizer_fit.transform(train_docs)

    # and convert to a list of (1,num_tokens) sparse matrices
    # we need to do this so we can set it as a column in the data frame
    document_tokens = [
        document_tokens[i] for i in range(document_tokens.shape[0])
    ]
    
    # for the demonstration, attach the identites back to the tokens
    document_tokens = {
        k: d for k, d in zip(sorted(documents.keys()), document_tokens)
    }

    # and create the data frame
    patient_df = pd.DataFrame.from_dict({
        'gender': gender,
        'race': race,
        'blood_pressure': blood_pressure,
        'heart_rate': heart_rate,
        'weight': weight,
        'document_tokens': document_tokens,
        'has_cm': has_cm,
        'document' : documents
    })
    patient_df = patient_df.reset_index().rename(columns={'index':'identity'})


    cols = ['blood_pressure', 'heart_rate']
    vals = [v for v in patient_df[cols].values.copy()]
    patient_df['heart_rate-blood_pressure'] = vals
    
    return patient_df

def to_bow_of_free_text(df, free_text_col):
    """
    Tokenizes free text into bag of words.
    """
    train_d = []
    for i in df[free_text_col]:
        train_d.append(i)

    train_d = np.asarray(train_d)

    # clean text
    train_d = np.array([clean_doc(d) for d in train_d])

    # tokenize text
    train_d = CountVectorizer().fit_transform(train_d)
    train_d = [train_d[i] for i in range(train_d.shape[0])]

    free_text = pd.Series(train_d)
    df[free_text_col] = free_text

def concatenate_scalar_columns(
        df:pd.DataFrame,
        scalar_column_names:Sequence[str]) -> Tuple[int, str]:
    new_col_name = '-'.join(scalar_column_names)
    float_vals = df[scalar_column_names].values.copy()
    
    df[new_col_name] = [
        fv for fv in float_vals
    ]
    
    column_index = df.columns.get_loc(new_col_name)
    return (column_index, new_col_name)

class SmallPatient():
    def __init__(self):
        g = load_small_patient_graph()
        # convert node id from alphabet to numbers
        g0 = nx.relabel_nodes(g, {k:i for i,k in enumerate([n for n in g])})

        data = from_networkx(g0)
        data.df = load_small_patient_data()

        # convert texts to bow
        for col in data.df.columns:
            if col == 'document' or col == 'gender' or col == 'race':
                to_bow_of_free_text(data.df, col)
        # concatenate float columns into a series
        list_of_float_single = ['blood_pressure', 'heart_rate', 'weight']
        count_in_list_of_float_single = len(list_of_float_single)

        indices, new_name = concatenate_scalar_columns(data.df,
                list_of_float_single)
        self.embedding_dim_float_singles = max(1,
                math.floor(math.log(count_in_list_of_float_single, 2)))
        data.allx = {col: data.df[col].values for col in data.df.columns}
        self.data = data

        list_of_features = ['identity', 'document_tokens', 'blood_pressure-heart_rate-weight']

if __name__ == '__main__':
    sp = SmallPatient()
    EP.cat_encoder(sp.data.allx['identity'])
    print('done')
