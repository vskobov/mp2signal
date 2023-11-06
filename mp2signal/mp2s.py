#!/usr/bin/env python
"""
Mediapipe To Signal Tool. 
Converts mediapipe holistic pose estimation into normalized 3d body series of joint's angles representation.
Normalizes the mediapipe pose in 3d space based on body proportions, and returns joint angles. 

Copyright (C) 2021-2023, Victor Skobov 
All rights reserved. E-mail: <vskobov@gmail.com>.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import mediapipe as mp
import joblib
import os
import cv2

### GLOBAL VARS ###
FACE_ANGLE = np.deg2rad(72)

mouth = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
outer_mouth = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
right_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
left_eye =  [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]
#right_brow =[247,30,29,27,28,56,190]
#left_brow =[414,286,258,257,259,260,467]
right_brow =[189,221,222,223,224,225]
left_brow = [413,441,442,443,444,445]

face_joints = mouth+outer_mouth+right_eye+right_brow+left_eye+left_brow
face_joints = [1001,1006] + list([j+1000 for j in face_joints])
#face_joints = [1001,1006,1000,1002,1003,1004,1005]+list([1000+k for k in range(7,468)])
MP_2_COCO_MAP = {
                '2':12,
                '3':14,
                '4':16,
                '5':11,
                '6':13,
                '7':15,
                '811':2423
                    }
                 
COCO_BODY_TREE = {  '1':[[2,5,811,999],0], # 'id' : [[children], distance to parent]
                    '2':[[3],0.5],
                    '3':[[4],0.86],
                    '4':[[400],0.8359*1.15],
                    '5':[[6],0.5],
                    '6':[[7],0.86],
                    '7':[[700],0.8359*1.15],
                    
                    '700': [[701, 705, 709, 713, 717], 0],
                    '400': [[401, 405, 409, 413, 417], 0],
                    
                    '701': [[702], 0.08268110413332652],
                    '702': [[703], 0.09961941283842532],
                    '703': [[704], 0.08202343020115353],
                    '704': [[], 0.06885685889325616],
                    '705': [[706], 0.2381580797638288],
                    '706': [[707], 0.1080568422968187],
                    '707': [[708], 0.0674891818042084],
                    '708': [[], 0.061922013487949054],
                    '709': [[710], 0.24202768593928028],
                    '710': [[711], 0.11665001533944559],
                    '711': [[712], 0.07430572174361719],
                    '712': [[], 0.06724562587722877],
                    '713': [[714], 0.23954279196590356],
                    '714': [[715], 0.10326888890666307],
                    '715': [[716], 0.06621362662226549],
                    '716': [[], 0.06062183423175105],
                    '717': [[718], 0.2327694272104333],
                    '718': [[719], 0.08467279556321614],
                    '719': [[720], 0.050094581739950565],
                    '720': [[], 0.04837138025087514],

                    '401': [[402], 0.08268110413332652],
                    '402': [[403], 0.09961941283842532],
                    '403': [[404], 0.08202343020115353],
                    '404': [[], 0.06885685889325616],
                    '405': [[406], 0.2381580797638288],
                    '406': [[407], 0.1080568422968187],
                    '407': [[408], 0.0674891818042084],
                    '408': [[], 0.061922013487949054],
                    '409': [[410], 0.24202768593928028],
                    '410': [[411], 0.11665001533944559],
                    '411': [[412], 0.07430572174361719],
                    '412': [[], 0.06724562587722877],
                    '413': [[414], 0.23954279196590356],
                    '414': [[415], 0.10326888890666307],
                    '415': [[416], 0.06621362662226549],
                    '416': [[], 0.06062183423175105],
                    '417': [[418], 0.2327694272104333],
                    '418': [[419], 0.08467279556321614],
                    '419': [[420], 0.050094581739950565],
                    '420': [[], 0.04837138025087514],
                    #'999': [[1001,1006,1000,1002,1003,1004,1005]+list([1000+k for k in range(7,468)]),(0.6831011424110601*0.6)], 
                    '811':[[],1.77],
                    '999': [face_joints,(0.6831011424110601*0.6)],
                    '1000': [[], 0.29897091464597064], '1001': [[], 0.32189085527721384], '1002': [[], 0.27952266914881957], '1003': [[], 0.2873609389947776], '1004': [[], 0.32392557213398476], '1005': [[], 0.31243649814727226], '1006': [[], 0.2650897399291037], '1007': [[], 0.23152953545672725], '1008': [[], 0.2663235758529026], '1009': [[], 0.28106111892961877], '1010': [[], 0.32792553789926154], '1011': [[], 0.3015310422076926], '1012': [[], 0.2996023972950409], '1013': [[], 0.293626352455436], '1014': [[], 0.2944574936262247], '1015': [[], 0.30208494805859454], '1016': [[], 0.31174384412318057], '1017': [[], 0.3166101657726286], '1018': [[], 0.31560574875389136], '1019': [[], 0.3146096926587482], '1020': [[], 0.2884030212559916], '1021': [[], 0.2695854636144719], '1022': [[], 0.21883150933608136], '1023': [[], 0.22479491506516333], '1024': [[], 0.22906646564494093], '1025': [[], 0.2325599631585523], '1026': [[], 0.21350079294133412], '1027': [[], 0.2487339835060737], '1028': [[], 0.24019499933543625], '1029': [[], 0.2510712940977683], '1030': [[], 0.24917886147850357], '1031': [[], 0.23377032666747213], '1032': [[], 0.3319746235340879], '1033': [[], 0.2326844686592723], '1034': [[], 0.2398845644874901], '1035': [[], 0.24020440793399275], '1036': [[], 0.24201256542682503], '1037': [[], 0.2956774702518987], '1038': [[], 0.2975211479118537], '1039': [[], 0.2917680441743019], '1040': [[], 0.28760126970423205], '1041': [[], 0.29360609925904346], '1042': [[], 0.2865694078069726], '1043': [[], 0.28911281372655506], '1044': [[], 0.3198155034654507], '1045': [[], 0.32145253886411734], '1046': [[], 0.27083555135383586], '1047': [[], 0.22916860567459177], '1048': [[], 0.2751797868331642], '1049': [[], 0.2666172174966195], '1050': [[], 0.2504157113580905], '1051': [[], 0.3067844617037169], '1052': [[], 0.27911597467424654], '1053': [[], 0.27835273131146177], '1054': [[], 0.289785851598389], '1055': [[], 0.26413907106306966], '1056': [[], 0.22755145740205635], '1057': [[], 0.27923622652293745], '1058': [[], 0.2572428166597987], '1059': [[], 0.26893831152838077], '1060': [[], 0.27124500175642363], '1061': [[], 0.2755012725190071], '1062': [[], 0.2758165225048924], '1063': [[], 0.2828162222134548], '1064': [[], 0.26695984038255915], '1065': [[], 0.2740534727318259], '1066': [[], 0.2861260657090454], '1067': [[], 0.32162954509347624], '1068': [[], 0.2853320573787695], '1069': [[], 0.3013965506524533], '1070': [[], 0.27130906765677215], '1071': [[], 0.26687583662006803], '1072': [[], 0.2995370013973689], '1073': [[], 0.2945062098271013], '1074': [[], 0.2885831308605261], '1075': [[], 0.26607287925282896], '1076': [[], 0.2760157205596095], '1077': [[], 0.2841103061607785], '1078': [[], 0.2750003234968581], '1079': [[], 0.2932260604809033], '1080': [[], 0.28226483576536127], '1081': [[], 0.2871572000251397], '1082': [[], 0.29181732605681954], '1083': [[], 0.31517114267394053], '1084': [[], 0.31523370928650346], '1085': [[], 0.3096005573149318], '1086': [[], 0.29974752691225426], '1087': [[], 0.29286920220168927], '1088': [[], 0.2840304016063443], '1089': [[], 0.2883318529746069], '1090': [[], 0.29409818204167026], '1091': [[], 0.296721165688954], '1092': [[], 0.27387342621783556], '1093': [[], 0.22845700495235619], '1094': [[], 0.29129372918385193], '1095': [[], 0.2788872712466281], '1096': [[], 0.2816799978728051], '1097': [[], 0.27468102371262265], '1098': [[], 0.25623893299599104], '1099': [[], 0.2735680441385208], '1100': [[], 0.22785582410862668], '1101': [[], 0.23580170263495484], '1102': [[], 0.25851722296418245], '1103': [[], 0.3088474137972539], '1104': [[], 0.2963441466183404], '1105': [[], 0.2864113952139477], '1106': [[], 0.29957896658144106], '1107': [[], 0.2832342677574787], '1108': [[], 0.3025243820931178], '1109': [[], 0.3274230513180256], '1110': [[], 0.23166032330255945], '1111': [[], 0.24004556319451353], '1112': [[], 0.2100483802698624], '1113': [[], 0.24927824973173995], '1114': [[], 0.23466065245157347], '1115': [[], 0.29049028521608267], '1116': [[], 0.24421449523156588], '1117': [[], 0.24094029128101563], '1118': [[], 0.23769349162062342], '1119': [[], 0.2262273003058249], '1120': [[], 0.21955960791112736], '1121': [[], 0.21980522605507588], '1122': [[], 0.2568960220709868], '1123': [[], 0.24726972681887469], '1124': [[], 0.2556920977528406], '1125': [[], 0.31319541428660225], '1126': [[], 0.23678883309142004], '1127': [[], 0.23350784163776783], '1128': [[], 0.22140415952525408], '1129': [[], 0.24028374230638613], '1130': [[], 0.23426403422232123], '1131': [[], 0.2802012698847483], '1132': [[], 0.23951076861897688], '1133': [[], 0.2078742927906687], '1134': [[], 0.2961852504944151], '1135': [[], 0.2909542723881395], '1136': [[], 0.29600314488739554], '1137': [[], 0.23569347155496254], '1138': [[], 0.27597316047149884], '1139': [[], 0.2489766370913873], '1140': [[], 0.34404294051549217], '1141': [[], 0.2895445914267923], '1142': [[], 0.2350230862840524], '1143': [[], 0.2425732853262155], '1144': [[], 0.2287705282735886], '1145': [[], 0.22520962923738644], '1146': [[], 0.285274288355511], '1147': [[], 0.2506409650747717], '1148': [[], 0.3596643057203916], '1149': [[], 0.3307656999912546], '1150': [[], 0.3149430588778669], '1151': [[], 0.3017147714050526], '1152': [[], 0.3628790585725905], '1153': [[], 0.22033872686503078], '1154': [[], 0.21444723794969936], '1155': [[], 0.20910977852901597], '1156': [[], 0.2557563695040086], '1157': [[], 0.2216385138802922], '1158': [[], 0.2314336909788791], '1159': [[], 0.23712780827823615], '1160': [[], 0.23999936721547363], '1161': [[], 0.23891048550538768], '1162': [[], 0.24956266804378788], '1163': [[], 0.23043039395196085], '1164': [[], 0.2806111128212048], '1165': [[], 0.2724869590362323], '1166': [[], 0.27552270659733225], '1167': [[], 0.280348223606196], '1168': [[], 0.25705077526428843], '1169': [[], 0.3075752459281116], '1170': [[], 0.3241387033429568], '1171': [[], 0.3600382920186313], '1172': [[], 0.2767319719208857], '1173': [[], 0.21343592319192303], '1174': [[], 0.2591518825694803], '1175': [[], 0.36278244905531715], '1176': [[], 0.348026388676393], '1177': [[], 0.24460732644538977], '1178': [[], 0.2885938155412595], '1179': [[], 0.2946149271579142], '1180': [[], 0.3023140223346934], '1181': [[], 0.30749741252221846], '1182': [[], 0.30972560805992644], '1183': [[], 0.28110481593877984], '1184': [[], 0.28202924303494115], '1185': [[], 0.2817372869433164], '1186': [[], 0.2762929911586161], '1187': [[], 0.2578028080213777], '1188': [[], 0.24379495939408302], '1189': [[], 0.22098866838439127], '1190': [[], 0.21643459265505352], '1191': [[], 0.2768000368782006], '1192': [[], 0.2666888170664094], '1193': [[], 0.24343028538942313], '1194': [[], 0.3187015088472006], '1195': [[], 0.29371007648092834], '1196': [[], 0.2728141910161798], '1197': [[], 0.2777078901733986], '1198': [[], 0.26033229354941745], '1199': [[], 0.35170744268583354], '1200': [[], 0.3313975253443943], '1201': [[], 0.3292121084295293], '1202': [[], 0.2928587899920742], '1203': [[], 0.24586044165649062], '1204': [[], 0.30667628699357635], '1205': [[], 0.25448286153764355], '1206': [[], 0.2572669318292015], '1207': [[], 0.26335195121591826], '1208': [[], 0.3486193720729546], '1209': [[], 0.2493398403150055], '1210': [[], 0.2988584653392856], '1211': [[], 0.3134480036082128], '1212': [[], 0.28119286789568776], '1213': [[], 0.25771405612829557], '1214': [[], 0.2812234899627166], '1215': [[], 0.25871284072526024], '1216': [[], 0.2690123554656391], '1217': [[], 0.24623383291847045], '1218': [[], 0.2965383034488502], '1219': [[], 0.27805815175841003], '1220': [[], 0.30683306952453265], '1221': [[], 0.23480325211637454], '1222': [[], 0.250337951878886], '1223': [[], 0.26024244049604595], '1224': [[], 0.2634564530389425], '1225': [[], 0.2603001463491855], '1226': [[], 0.23713123418566742], '1227': [[], 0.23580978844465622], '1228': [[], 0.23213927781261923], '1229': [[], 0.22909700291366253], '1230': [[], 0.2229519321148801], '1231': [[], 0.21636136284150032], '1232': [[], 0.21318985382648759], '1233': [[], 0.21257065771194214], '1234': [[], 0.22575207017692347], '1235': [[], 0.27014205189151563], '1236': [[], 0.27418133020401136], '1237': [[], 0.3105043812594327], '1238': [[], 0.30439628303041194], '1239': [[], 0.30198493649277774], '1240': [[], 0.2646309991244043], '1241': [[], 0.3099498109173028], '1242': [[], 0.28919633095377734], '1243': [[], 0.2082312669669611], '1244': [[], 0.21577909023291397], '1245': [[], 0.2280737421976012], '1246': [[], 0.23675017303874524], '1247': [[], 0.24417045278698996], '1248': [[], 0.2874446463997888], '1249': [[], 0.2348494014643371], '1250': [[], 0.28947859846506585], '1251': [[], 0.2705817007354643], '1252': [[], 0.219107737448443], '1253': [[], 0.22544308950484077], '1254': [[], 0.23062297252307387], '1255': [[], 0.2349356351177401], '1256': [[], 0.21431886232087952], '1257': [[], 0.2504244453392782], '1258': [[], 0.24215013518115627], '1259': [[], 0.2533521865211508], '1260': [[], 0.2517713284546768], '1261': [[], 0.23643112691448845], '1262': [[], 0.33062123936508103], '1263': [[], 0.2360954653987609], '1264': [[], 0.24012154849462752], '1265': [[], 0.242751797672941], '1266': [[], 0.24490194995628234], '1267': [[], 0.2959856206347846], '1268': [[], 0.29737436607679113], '1269': [[], 0.2917981411842897], '1270': [[], 0.28705146430049505], '1271': [[], 0.2931173799895994], '1272': [[], 0.28593242938586805], '1273': [[], 0.2885315227701197], '1274': [[], 0.3207875369708608], '1275': [[], 0.32254351873265463], '1276': [[], 0.2739254502249646], '1277': [[], 0.23068910582925026], '1278': [[], 0.27741370734206716], '1279': [[], 0.2689262724794815], '1280': [[], 0.25170081718439397], '1281': [[], 0.3071107394652414], '1282': [[], 0.2831348033070644], '1283': [[], 0.282146263910447], '1284': [[], 0.2917777827545273], '1285': [[], 0.26534728964165333], '1286': [[], 0.22922964683725006], '1287': [[], 0.2784687265266037], '1288': [[], 0.251218580811986], '1289': [[], 0.2711917427787328], '1290': [[], 0.27387898222713714], '1291': [[], 0.27447771124293135], '1292': [[], 0.27472644918399874], '1293': [[], 0.2869431371915004], '1294': [[], 0.2695339254082094], '1295': [[], 0.27744151725851635], '1296': [[], 0.2897415812540241], '1297': [[], 0.3239024273433314], '1298': [[], 0.28860223719336375], '1299': [[], 0.30455446095312577], '1300': [[], 0.27459714009256875], '1301': [[], 0.26940260677717154], '1302': [[], 0.29943949487346855], '1303': [[], 0.29412370901344387], '1304': [[], 0.2881410611710916], '1305': [[], 0.26857118616154846], '1306': [[], 0.27506063809348635], '1307': [[], 0.283771875800349], '1308': [[], 0.273787070112275], '1309': [[], 0.29389104696010604], '1310': [[], 0.28141732482869725], '1311': [[], 0.28655662923730263], '1312': [[], 0.29143372873223444], '1313': [[], 0.3153728304957644], '1314': [[], 0.3154737063906793], '1315': [[], 0.3096976198995524], '1316': [[], 0.2996960390688427], '1317': [[], 0.29277202838770633], '1318': [[], 0.2837659364877235], '1319': [[], 0.2884397646044949], '1320': [[], 0.2943714836070328], '1321': [[], 0.2967900879787643], '1322': [[], 0.2750049806639789], '1323': [[], 0.22748199804569852], '1324': [[], 0.2779328925773773], '1325': [[], 0.2811695648059718], '1326': [[], 0.27611957061871806], '1327': [[], 0.25971229531627865], '1328': [[], 0.2756500763928238], '1329': [[], 0.22967913050936326], '1330': [[], 0.2382105080627384], '1331': [[], 0.261172940793423], '1332': [[], 0.31154034713941386], '1333': [[], 0.30036532735425386], '1334': [[], 0.29072427694376946], '1335': [[], 0.29883619337752215], '1336': [[], 0.2849459552210664], '1337': [[], 0.304423473688559], '1338': [[], 0.32949216621720046], '1339': [[], 0.23368154672991998], '1340': [[], 0.24204957711435982], '1341': [[], 0.21062513234064104], '1342': [[], 0.2520229182200012], '1343': [[], 0.23574325629364182], '1344': [[], 0.2923603926771835], '1345': [[], 0.24436646164204984], '1346': [[], 0.24268761940403513], '1347': [[], 0.2399083134158251], '1348': [[], 0.22810700436716558], '1349': [[], 0.22101851576726902], '1350': [[], 0.22119905543344628], '1351': [[], 0.257425921901319], '1352': [[], 0.24641525578227047], '1353': [[], 0.2580102193904131], '1354': [[], 0.3134093596614719], '1355': [[], 0.2384615874858642], '1356': [[], 0.23413796800684117], '1357': [[], 0.222372713169181], '1358': [[], 0.24346538433889683], '1359': [[], 0.2371971586259149], '1360': [[], 0.2821946594493414], '1361': [[], 0.23603419600413414], '1362': [[], 0.20838279855191433], '1363': [[], 0.2972867211456871], '1364': [[], 0.2858816963777874], '1365': [[], 0.28945461209031653], '1366': [[], 0.23444087835678357], '1367': [[], 0.27035171628555116], '1368': [[], 0.25097438981294534], '1369': [[], 0.34188401292123033], '1370': [[], 0.2901846699657251], '1371': [[], 0.23736400028037194], '1372': [[], 0.24381235560577538], '1373': [[], 0.23106403176080104], '1374': [[], 0.22633272875410823], '1375': [[], 0.284910319057354], '1376': [[], 0.24883362482984367], '1377': [[], 0.3584493087504406], '1378': [[], 0.3269506069880008], '1379': [[], 0.3095011812155092], '1380': [[], 0.22107373959865928], '1381': [[], 0.2147661372310292], '1382': [[], 0.20938255376225964], '1383': [[], 0.25783370062986893], '1384': [[], 0.22211272164633708], '1385': [[], 0.23229813659866475], '1386': [[], 0.23857589829793993], '1387': [[], 0.24217663670152922], '1388': [[], 0.24167681437930302], '1389': [[], 0.2508608089676983], '1390': [[], 0.23327237906123913], '1391': [[], 0.2742620263157011], '1392': [[], 0.2779549327368845], '1393': [[], 0.2813181973523481], '1394': [[], 0.3034663564644981], '1395': [[], 0.320811971882683], '1396': [[], 0.35908329812836315], '1397': [[], 0.2697498391055725], '1398': [[], 0.21373061929278836], '1399': [[], 0.26016747142932367], '1400': [[], 0.34527493898846445], '1401': [[], 0.24121088138170843], '1402': [[], 0.2885424736978461], '1403': [[], 0.294681967932122], '1404': [[], 0.3023251462419121], '1405': [[], 0.30739146457514877], '1406': [[], 0.30924496937181445], '1407': [[], 0.2801751057964011], '1408': [[], 0.2810558917151956], '1409': [[], 0.28071135385486734], '1410': [[], 0.27623975033744225], '1411': [[], 0.25709783753793486], '1412': [[], 0.24476497334601438], '1413': [[], 0.22233451176825544], '1414': [[], 0.21692884955489472], '1415': [[], 0.27572722830198854], '1416': [[], 0.2637237801083085], '1417': [[], 0.24420644188841553], '1418': [[], 0.31769238776362274], '1419': [[], 0.2736160203163983], '1420': [[], 0.26222317582711624], '1421': [[], 0.32905424949402307], '1422': [[], 0.2916197933371307], '1423': [[], 0.2492153573085178], '1424': [[], 0.30535801299583243], '1425': [[], 0.25696518540036645], '1426': [[], 0.25967093828610327], '1427': [[], 0.2640436167441093], '1428': [[], 0.3480509768734033], '1429': [[], 0.2514668171565071], '1430': [[], 0.29638142996672173], '1431': [[], 0.3114428151234808], '1432': [[], 0.28007653993447296], '1433': [[], 0.25481993957879207], '1434': [[], 0.27870070807229896], '1435': [[], 0.2535965233052469], '1436': [[], 0.2698085719527195], '1437': [[], 0.24740365827134572], '1438': [[], 0.297683845338127], '1439': [[], 0.28067341512352934], '1440': [[], 0.30829789678497377], '1441': [[], 0.23628249569638507], '1442': [[], 0.2529013338797679], '1443': [[], 0.263330956428856], '1444': [[], 0.2666212441678272], '1445': [[], 0.26320590977526453], '1446': [[], 0.24004215219791578], '1447': [[], 0.2352552097879248], '1448': [[], 0.23464258090368315], '1449': [[], 0.23134207934867287], '1450': [[], 0.22451091736110804], '1451': [[], 0.2174721939326344], '1452': [[], 0.21460846419653667], '1453': [[], 0.21375967260725265], '1454': [[], 0.22575207017692347], '1455': [[], 0.2727684228920949], '1456': [[], 0.27503637528716735], '1457': [[], 0.3118353234239305], '1458': [[], 0.30550031486301726], '1459': [[], 0.30308276335110307], '1460': [[], 0.267713336114065], '1461': [[], 0.3107750080147737], '1462': [[], 0.2901356448738235], '1463': [[], 0.2092754139355229], '1464': [[], 0.21687201932915684], '1465': [[], 0.22911971988064916], '1466': [[], 0.23979174819467683], '1467': [[], 0.24730243642198935]}

base_dir = os.path.dirname(__file__)
RELAXED_FACE_MODEL = os.path.join(base_dir,'relaxed_face.joblib')

### CLASSES ###
class Joint_Tree:
    def __init__(self, Body_Dict=COCO_BODY_TREE, id=1, parent=None):
        self.id = id
        if parent:
            self.level = parent.level+1
        else:
            self.level = 0
            self.relaxed_face = joblib.load(RELAXED_FACE_MODEL)
        self.parent = parent
        self.BT = Body_Dict
        self.children = list([Joint_Tree(Body_Dict,int(c),self) for c in Body_Dict[str(self.id)][0]])
        self.joint_datum = []        
        self.dist_to_parent = Body_Dict[str(self.id)][1]

    def _get_all_joints(self):
        #preorder iterration
        j_list = [self]
        if self.is_leaf() == False:
            for i in range(len(self.children)):
                j_list = j_list + self.children[i]._get_all_joints()
        return j_list

    def __iter__(self):   
        all_joints = iter(self._get_all_joints())
        return all_joints

    def _mp_pose_convert(self, id):
        return int(MP_2_COCO_MAP[str(id)])

    def process(self, sign_mov, b_mul = None):
        #print('Process Tr',self.id)
        #if sign_mov['Meta']:
        #    self.meta = sign_mov['Meta']
        if hasattr(self,'normed_coords'):
            delattr(self,'normed_coords')
            delattr(self,'rotated_coords')
            delattr(self,'basic_normalization_coords')

            delattr(self,'_alpha')
            delattr(self,'_beta')
            delattr(self,'_gamma')
            delattr(self,'rel_a')


        if self.parent:
            self.last_joint_id = self.parent.last_joint_id
        else:
            self.last_joint_id = int(list(j.id for j in self)[-1])

        if self.is_root(): # '1" point as origin
            if b_mul:
                self._get_root().body_muls = np.array([b_mul])
            
            if len(sign_mov['MP_Pose'])==1:
                if sign_mov['MP_Face'].shape!=(0,):
                    new_body_muls = self._get_body_multiplier(sign_mov['MP_Face'])
                    if new_body_muls[0]>1:
                        if hasattr(self,'body_muls')==False:
                            self.body_muls=new_body_muls
                        else:
                            if abs(self.body_muls[0]-new_body_muls[0])>3:
                                self.body_muls= new_body_muls

                if hasattr(self,'body_muls')==False:
                    self.body_muls=np.array([98.0])
            else:
               self.body_muls = self._get_body_multiplier(sign_mov['MP_Face'])

            self.joint_datum = self._get_datum(sign_mov['MP_Pose'])
            
        elif self.id >= 400 and self.id < 500:
            #RIGHT HAND
            if sign_mov['MP_RHand'].shape!=(0,):
                self.joint_datum = self._get_datum(sign_mov['MP_RHand'],400)
        elif self.id >= 700 and self.id < 800:
            #LEFT HAND
            if sign_mov['MP_LHand'].shape!=(0,):
                self.joint_datum = self._get_datum(sign_mov['MP_LHand'],700)
        elif self.id >= 999:
            #FACE 
            if sign_mov['MP_Face'].shape!=(0,):
                self.joint_datum = self._get_datum(sign_mov['MP_Face'],1000,additional_datum=sign_mov['MP_Pose'])
        else:
            self.joint_datum = self._get_datum(sign_mov['MP_Pose'])

        if self.is_leaf() == False:
            for i in range(len(self.children)):
                self.children[i].process(sign_mov,b_mul)
        else:
            if self.id==self.last_joint_id:
                self._get_root().shoulder_distance_for_basic_normalization = a_distance_two_points_3d(self._get_root()._get_joint(2).joint_datum,self._get_root()._get_joint(5).joint_datum)
                self._get_root().normalize_recursive()
            return

    def from_gram_process(self,gram):
        joints = list(k for k in self)
        r = gram.shape[1]-(len(joints)*3)
        for i in range(len(joints)):
            j = joints[i]
            _alpha = np.zeros((gram.shape[1]))
            _beta = np.zeros((gram.shape[1]))
            _gamma = np.zeros((gram.shape[1]))
            rel_a = np.zeros((gram.shape[1]))
            rotated_coords = np.zeros((gram.shape[1],3))
            if j.dist_to_parent!=0:

                rel_a = uint8_to_angle(gram[i])

                _alpha = uint8_to_angle(gram[r+(i*1)])
                _beta  = uint8_to_angle(gram[r+(i*2)])
                _gamma = uint8_to_angle(gram[r+(i*3)])

                l_i =  j.dist_to_parent * 100
                x_i = np.cos(_alpha)/l_i
                y_i = np.cos(_beta)/l_i
                z_i = np.cos(_gamma)/l_i
                rotated_coords[:,0] = j.parent.rotated_coords[:,0] + x_i
                rotated_coords[:,1] = j.parent.rotated_coords[:,1] + y_i
                rotated_coords[:,2] = j.parent.rotated_coords[:,2] + z_i
            j._alpha = _alpha
            j._beta = _beta
            j._gamma = _gamma
            j.rotated_coords = rotated_coords
            j.rel_a = rel_a
        return

    def normalize_recursive(self):
        #print('Norm Rec Tr',self.id)
        if self.is_root(): # '1" point as origin
            self.normed_coords = np.zeros((self.joint_datum.shape[0],4))
            self.normed_coords[:,-1] = np.ones(self.normed_coords.shape[0])
            self.basic_normalization_coords = np.zeros((self.joint_datum.shape[0],3))
        elif self.id >= 400 and self.id < 500:
            #RIGHT HAND
            self._get_normed_datum_coords_hand(ancestor=4)
            self._get_basic_normalization_coords()
        elif self.id >= 700 and self.id < 800:
            #LEFT HAND
            self._get_normed_datum_coords_hand(ancestor=7)
            self._get_basic_normalization_coords()
        elif self.id >= 999:
            #FACE 
            self._get_normed_datum_coords_face()
            self._get_basic_normalization_coords()
        else:
            self._get_normed_datum_coords()
            self._get_basic_normalization_coords()

        if self.is_leaf() == False:
            for i in range(len(self.children)):
                self.children[i].normalize_recursive()
        else:
            if self.id==self.last_joint_id:
                self._get_root().rotate_recursive()
            return

    def rotate_recursive(self):
        #print('Rotate Tr',self.id)
        self._alpha = np.zeros((len(self.joint_datum)))
        self._beta = np.zeros((len(self.joint_datum)))
        self._gamma = np.zeros((len(self.joint_datum)))
        self.rel_a = np.zeros((self.joint_datum.shape[0]))
        if self.is_root(): # '1" point as origin
            self.rotated_coords = np.zeros((self.joint_datum.shape[0],3))
            self.color = np.vstack((angle_to_uint8(self._alpha),angle_to_uint8(self._beta),angle_to_uint8(self._gamma))).T
            self.signal = np.vstack((self._alpha,self._beta,self._gamma)).T
            self.relative_color = angle_to_uint8(self.rel_a)
        elif self.id >= 400 and self.id < 500:
            #RIGHT HAND
            self._h_transform_coords(4,400)

        elif self.id >= 700 and self.id < 800:
            #LEFT HAND
            self._h_transform_coords(7,700)

        elif self.id > 999:
            #FACE 
            self._f_transform_coords()
        else:
            self._transform_coords()

        if self.is_leaf() == False:
            for i in range(len(self.children)):
                self.children[i].rotate_recursive()
        else:
            return

    def is_ancestor(self, id):
        if self.is_root()==False:
            if self.parent.id == id:
                return True
            else: 
                return self.parent.is_ancestor(id)
        else: 
            return False

    def __getitem__(self,id):
        return self._get_root()._get_joint(id)

    def is_root(self):
        if self.parent == None:
            return True
        else:
            return False
            
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False
            
    def __str__(self):
        return str(self.id) + ' Children: ' + str([x.id for x in self.children])

    def _add_child(self, JV):
        assert isinstance(JV, Joint_Tree)
        self.adjacent.append(JV)

    def _get_id(self):
        return self.id

    def _get_joint(self, id):

        if self.id == id: 
            #print('Found ',id,' in ', self.parent.id)
            return self

        res = None
        for c in self.children:
            res = c._get_joint(id)
            if res != None: 
                return res

    def _get_root(self):
        if self.is_root()==True: 
            return self 
        else: 
            return self.parent._get_root()            
    
    def _get_body_multiplier(self, face_datum):

        l_eye_center = np.average((face_datum[:,159],face_datum[:,143],face_datum[:,157],face_datum[:,149]),axis=0)
        r_eye_center = np.average((face_datum[:,384],face_datum[:,386],face_datum[:,379],face_datum[:,372]),axis=0)
        eye_distance = a_distance_two_points_3d(l_eye_center,r_eye_center)
        
        
        #eye_ratio = 0.15
        eye_ratio = 0.237
        bm = eye_distance / eye_ratio
        if len(bm)>1:
            intr = np.interp(np.argwhere(bm<1).reshape(-1),np.argwhere(bm>=1).reshape(-1),bm[bm>=1])
            bm[np.argwhere(bm<1).reshape(-1)] = intr

        '''
        c=0
        summ = 0
        for i in range(len(eye_distance)):
            if eye_distance[i]>1:
                summ += eye_distance[i]
                c+=1
        bm = np.zeros(eye_distance.shape)
        multiplier = (summ/c)/0.237
        bm.fill(multiplier)
        '''
        return bm

    def _get_datum(self, datum, mediator=0, additional_datum=[]):      
        _or_datum = np.zeros(datum.shape)
        if self.is_root()==False:
            if mediator==0:
                _mp_id= self._mp_pose_convert(self.id)
                #CENTER OF THE HIP
                if _mp_id==2423:
                    _or_datum = np.average((datum[:,24],datum[:,23]),axis = 0)
                else:
                    #ALL BODY LANDMARKS
                    _or_datum = datum[:,_mp_id]
            elif mediator==1000:
                _or_datum_pose = np.average((additional_datum[:,8],additional_datum[:,7]),axis=0)
                face_center = np.average((datum[:,234],datum[:,454]),axis=0)
                if self.id == 999:
                    #HEAD CENTER
                    #_or_datum = face_center[:]
                    _or_datum = _or_datum_pose[:]
                    #_or_datum[:,2] = _or_datum_pose[:,2]
                else:
                    #FACE LANDMARKS
                    _or_datum = datum[:,self.id-mediator] - face_center[:,:4] + _or_datum_pose[:,:4]
            else:
                #HAND LANDMARKS
                wrist_base = np.average((datum[:,0],datum[:,1]),axis=0)
                #if self.id == 706 or self.id==406:
                #    print(self.id,self._get_root()._get_joint(int(mediator/100)).joint_datum[:,:4],datum[:,self.id-mediator] )
                _or_datum = datum[:,self.id-mediator] - wrist_base + self._get_root()._get_joint(int(mediator/100)).joint_datum[:,:4]
        else:
            #ROOT CENTER OF THE SHOULDERS
            _or_datum = np.average((datum[:,11],datum[:,12]),axis = 0)
            
        return _or_datum

    def _get_normed_datum_coords_face(self):
        
        x_i = self.joint_datum[:,0]-self.parent.joint_datum[:,0]
        y_i = self.joint_datum[:,1]-self.parent.joint_datum[:,1]
        z_i = self.joint_datum[:,2]-self.parent.joint_datum[:,2]

        x_i = x_i if len(x_i)==1 else interpolate_zeros(x_i)
        y_i = y_i if len(y_i)==1 else interpolate_zeros(y_i)  
        z_i = z_i if len(z_i)==1 else interpolate_zeros(z_i)  

        x_i = x_i + self.parent.normed_coords[:,0] 
        y_i = y_i + self.parent.normed_coords[:,1]   
        z_i = z_i + self.parent.normed_coords[:,2]   

        if self.id==999 :
            z_i = z_i * 0
    
        self.normed_coords = np.vstack((x_i,y_i,z_i,np.ones((x_i.shape[0])))).T
        #print('Tr',self.id)
        return
  
    def _get_normed_datum_coords_hand_basic(self, ancestor=4):
        
        l_i = self.dist_to_parent * self._get_root().body_muls
        x_i = self.joint_datum[:,0]-self.parent.joint_datum[:,0]
        y_i = self.joint_datum[:,1]-self.parent.joint_datum[:,1]
        or_3d = a_distance_two_points_3d(self.joint_datum,self.parent.joint_datum)
        sin_x = x_i/or_3d
        sin_y = y_i/or_3d
        x_i = l_i * sin_x
        y_i = l_i * sin_y
        z_i = np.power(abs(np.power(l_i,2)-np.power(x_i,2)-np.power(y_i,2)),0.5)
        _z_i = self.joint_datum[:,2]-self._get_root()._get_joint(ancestor).joint_datum[:,2]
        for i in range(len(z_i)): 
            if _z_i[i]<0:
                z_i[i] = -z_i[i]
        
        x_i = x_i if len(x_i)==1 else smooth_out(x_i,5,1)
        y_i = y_i if len(y_i)==1 else smooth_out(y_i,5,1)
        z_i = z_i if len(z_i)==1 else smooth_out(z_i,5,1)

        x_i = x_i + self.parent.normed_coords[:,0]   
        y_i = y_i + self.parent.normed_coords[:,1]   
        z_i = z_i + self.parent.normed_coords[:,2]   

        self.normed_coords = np.vstack((x_i,y_i,z_i,np.ones((x_i.shape[0])))).T
        return

    def _get_normed_datum_coords_hand(self, ancestor=4):
        
        if self.id == (ancestor*100):
            self._get_normed_datum_coords_hand_basic(ancestor)
            self._get_joint(ancestor*100+5)._get_normed_datum_coords_hand_basic(ancestor)
            self._get_joint(ancestor*100+9)._get_normed_datum_coords_hand_basic(ancestor)
            self._get_joint(ancestor*100+13)._get_normed_datum_coords_hand_basic(ancestor)
            self._get_joint(ancestor*100+17)._get_normed_datum_coords_hand_basic(ancestor)
            p0 = self._get_root()._get_joint(ancestor*100+0).normed_coords[:,:3]
            p1 = self._get_root()._get_joint(ancestor*100+5).normed_coords[:,:3]
            p2 = self._get_root()._get_joint(ancestor*100+17).normed_coords[:,:3]
            u = p1-p0
            v = p2-p0

            u_cross_v = np.zeros((p0.shape[0],3),dtype=float)
            for i in range(p0.shape[0]):
                u_cross_v[i] = np.cross(u[i],v[i])

            u_cross_v = u_cross_v

            #checking on the middle finger
            c_id = ancestor*100 +10
            l_i = self[c_id].dist_to_parent * self[c_id]._get_root().body_muls
            x_i = self[c_id].joint_datum[:,0]-self[c_id].parent.joint_datum[:,0]
            y_i = self[c_id].joint_datum[:,1]-self[c_id].parent.joint_datum[:,1]
            
            or_3d = a_distance_two_points_3d(self[c_id].joint_datum,self[c_id].parent.joint_datum)
            sin_x = x_i/or_3d
            sin_y = y_i/or_3d
            x_i = l_i * sin_x
            y_i = l_i * sin_y
            z_i = np.power(abs(np.power(l_i,2)-np.power(x_i,2)-np.power(y_i,2)),0.5)

            point  = np.vstack((x_i,y_i,z_i)).T

            _z_i = np.zeros(point.shape[0])
            _d = np.ones(point.shape[0])
            

            for i in range(point.shape[0]):
                if  ancestor==4:
                    _z_i[i] = - np.dot(point[i],u_cross_v[i])    
                else:
                    _z_i[i] = np.dot(point[i],u_cross_v[i])

            for i in range(len(z_i)): 
                if _z_i[i]<0:
                    z_i[i] = -z_i[i]
                    _d[i] = -_d[i]
            '''

            #Cheking angain if hands where dublicated and mirroring back

            _dd = np.ones(point.shape[0])
            _z_i = np.zeros(point.shape[0])

            point  = np.vstack((x_i,y_i,z_i)).T  
            for i in range(point.shape[0]):
                    if  ancestor==4:
                        _z_i[i] = - np.dot(point[i],u_cross_v[i])    
                    else:
                        _z_i[i] = np.dot(point[i],u_cross_v[i])
                        
            for i in range(len(z_i)): 
                if _z_i[i]<0:
                    _dd[i] = -_dd[i]
            
            self._dd = _dd
            '''

            self._d = _d
            
        else:            
            if hasattr(self,'normed_coords')==False:
                l_i =  self.dist_to_parent * self._get_root().body_muls
                x_i = self.joint_datum[:,0]-self.parent.joint_datum[:,0]
                y_i = self.joint_datum[:,1]-self.parent.joint_datum[:,1]
                or_3d = a_distance_two_points_3d(self.joint_datum,self.parent.joint_datum)
                sin_x = x_i/or_3d
                sin_y = y_i/or_3d
                x_i = l_i * sin_x
                y_i = l_i * sin_y
                z_i = np.power(abs(np.power(l_i,2)-np.power(x_i,2)-np.power(y_i,2)),0.5)

                z_i = z_i * self._get_root()._get_joint(ancestor*100)._d
                #x_i = x_i * self._get_root()._get_joint(ancestor*100)._dd

                x_i = smooth_out(x_i,5,1)
                y_i = smooth_out(y_i,5,1)
                z_i = smooth_out(z_i,5,1)

                x_i = x_i + self.parent.normed_coords[:,0]   
                y_i = y_i + self.parent.normed_coords[:,1]   
                z_i = z_i + self.parent.normed_coords[:,2]   

                self.normed_coords = np.vstack((x_i,y_i,z_i,np.ones((x_i.shape[0])))).T
                #print('Tr',self.id)
            return

    def _get_normed_datum_coords(self):
            
        l_i =  self.dist_to_parent * self._get_root().body_muls
        x_i = self.joint_datum[:,0]-self.parent.joint_datum[:,0]
        y_i = self.joint_datum[:,1]-self.parent.joint_datum[:,1]
        z_i = np.power(abs(np.power(l_i,2)-np.power(x_i,2)-np.power(y_i,2)),0.5)
        
        if self.id==3 or self.id==6:
            _z_i = self.joint_datum[:,2]
        else:
            _z_i = (self.joint_datum[:,2]-self.parent.joint_datum[:,2])

        for i in range(len(z_i)): 
            if _z_i[i]<0:
                z_i[i] = -z_i[i]
        
        x_i = x_i if len(x_i)==1 else smooth_out(x_i,5,1)
        y_i = y_i if len(y_i)==1 else smooth_out(y_i,5,1)
        z_i = z_i if len(z_i)==1 else smooth_out(z_i,5,1)

        x_i = x_i + self.parent.normed_coords[:,0]   
        y_i = y_i + self.parent.normed_coords[:,1]   
        z_i = z_i + self.parent.normed_coords[:,2]   

    
        self.normed_coords = np.vstack((x_i,y_i,z_i,np.ones((x_i.shape[0])))).T
        #print('Tr',self.id)
        return

    def _get_basic_normalization_coords(self):
            
        x_i = (self.joint_datum[:,0]-self._get_root().joint_datum[:,0]) / self._get_root().shoulder_distance_for_basic_normalization
        y_i = (self.joint_datum[:,1]-self._get_root().joint_datum[:,1]) / self._get_root().shoulder_distance_for_basic_normalization
        z_i = (self.joint_datum[:,2]-self._get_root().joint_datum[:,2]) / self._get_root().shoulder_distance_for_basic_normalization

        #x_i = smooth_out(x_i,5,1)
        #y_i = smooth_out(y_i,5,1)
        #z_i = smooth_out(z_i,5,1)

        self.basic_normalization_coords = np.vstack((x_i,y_i,z_i)).T
        #print('Tr',self.id)
        return

    def _get_rotated_coords(self, body_ancestor=0):
        if body_ancestor==0:
            target_coords = np.zeros(self.normed_coords.shape)

            if self.id == 2:
                target_coords[:,0] = -self.dist_to_parent * self._get_root().body_muls

            elif self.id == 5:
                target_coords[:,0] = self.dist_to_parent * self._get_root().body_muls

            target_coords[:,-1] = np.ones(target_coords.shape[0])
            rotated_coords = np.zeros((target_coords.shape[0],3))
            scale_coeffs = np.zeros((target_coords.shape[0]))
            rotation_mtrxs = np.zeros((target_coords.shape[0],4,4))
            for i in range(target_coords.shape[0]):
                rotation_mtrxs[i] = get_rotation_mat(self.normed_coords[i], target_coords[i])
                rotated_c = rotate(self.normed_coords[i],rotation_mtrxs[i])
                scale_coeffs[i] = abs(100/(rotated_c[0]+0.000001))
                rotated_coords[i] = scale(rotated_c,scale_coeffs[i])

            self.rotated_coords = rotated_coords
            self.rotation_mtrxs = rotation_mtrxs
            self.scale_coeffs = scale_coeffs

            return 
        else:    

            rotated_coords = np.zeros((self.normed_coords.shape[0],3))
            rotation_mtrxs = self._get_root()._get_joint(body_ancestor).rotation_mtrxs
            #not efficient
            #rotated_c = nrotate(self.normed_coords,self._get_root()._get_joint(body_ancestor).rotation_mtrxs)
            #self.rotated_coords =nscale(rotated_c,self._get_root()._get_joint(body_ancestor).scale_coeffs)

            for i in range(self.normed_coords.shape[0]):
                rotated_c = rotate(self.normed_coords[i],rotation_mtrxs[i])
                rotated_coords[i] = scale(rotated_c,self._get_root()._get_joint(body_ancestor).scale_coeffs[i])

            self.rotated_coords = rotated_coords

            return 

    def _get_face_rotated_coords(self):
        #print('Tr ', self.id)
        self.head_origin_normed= self.normed_coords - self.parent.normed_coords
        self.head_origin_normed[:,-1] = np.ones(self.head_origin_normed.shape[0])

        if self.id==1001:
            target_coords = np.zeros(self.head_origin_normed.shape)
            target_coords[:,2] = self.dist_to_parent * self._get_root().body_muls
            target_coords[:,-1] = np.ones(target_coords.shape[0])

            rotated_coords = np.zeros((target_coords.shape[0],3))
            scale_coeffs = np.zeros((target_coords.shape[0]))
            rotation_mtrxs = np.zeros((target_coords.shape[0],4,4))

            face_normed_1006_coords = self[1006].normed_coords - self[1006].parent.normed_coords
            face_normed_1006_coords[:,-1] = np.ones(self.head_origin_normed.shape[0])

            for i in range(target_coords.shape[0]):
                rotation_mtrxs[i] = get_face_rotation_mat(self.head_origin_normed[i], target_coords[i],face_normed_1006_coords[i],np.array([0,-1,0,1]))
                
                rotated_c = rotate(self.head_origin_normed[i],rotation_mtrxs[i])
                scale_coeffs[i] = abs(150/(rotated_c[2]+0.000001))
                rotated_coords[i] = scale(rotated_c,scale_coeffs[i])

            #print(target_coords,rotated_coords,scale_coeffs)

            self.face_rotated_coords = rotated_coords
            self.face_rotation_mtrxs = rotation_mtrxs
            self.face_scale_coeffs = scale_coeffs

            return 
        else:    

            rotated_coords = np.zeros((self.head_origin_normed.shape[0],3))
            rotation_mtrxs = self._get_root()._get_joint(1001).face_rotation_mtrxs

            for i in range(self.head_origin_normed.shape[0]):
                rotated_c = rotate(self.head_origin_normed[i],rotation_mtrxs[i])
                rotated_coords[i] = scale(rotated_c,self._get_root()._get_joint(1001).face_scale_coeffs[i])

            self.face_rotated_coords = rotated_coords

            return 

    def _transform_coords(self):
        if self.is_root(): # '1" point as origin
            self.rotated_coords = np.zeros((self.joint_datum.shape[0],3))
            return

        if hasattr(self,'normed_coords'):
            #print('Tr',self.id)
            if self.id==2 or self.id==5:
                self._get_rotated_coords()
            else:
                if self.is_ancestor(5):
                    self._get_rotated_coords(5)
                elif self.is_ancestor(2):
                    self._get_rotated_coords(2)
                else:
                    self.rotated_coords = np.zeros((self.joint_datum.shape[0],3))
                    self.rotated_coords[:,1] = self.dist_to_parent * self._get_root().body_muls
        
            x_i = self.rotated_coords[:,0]-self.parent.rotated_coords[:,0]
            y_i = self.rotated_coords[:,1]-self.parent.rotated_coords[:,1]
            z_i = self.rotated_coords[:,2]-self.parent.rotated_coords[:,2]
            l_i = (x_i**2 + y_i**2 + z_i**2)**0.5 + 0.000001
            
            self._alpha = (np.arccos(x_i/l_i))
            self._beta =  (np.arccos(y_i/l_i))
            self._gamma = (np.arccos(z_i/l_i))

            if self.level >= 3:    
                rel_a = np.zeros(self.rotated_coords.shape[0])
                for i in range(self.rotated_coords.shape[0]):
                    rel_a[i] = angle_rad((self.rotated_coords[i]-self.parent.rotated_coords[i]),(self.parent.parent.rotated_coords[i]-self.parent.rotated_coords[i]))
                self.rel_a = rel_a
                self.relative_color = relative_angle_to_RGB_uint8(self.rel_a)

            self.color = np.vstack((angle_to_uint8(self._alpha),angle_to_uint8(self._beta),angle_to_uint8(self._gamma))).T
            self.signal = np.vstack((self._alpha,self._beta,self._gamma)).T

        return

    def _h_transform_coords(self, body_parent, mediator):
        if hasattr(self,'normed_coords'):

            if self.is_ancestor(2):
                self._get_rotated_coords(2)
            elif self.is_ancestor(5):
                self._get_rotated_coords(5)

            x_i = self.rotated_coords[:,0]-self.parent.rotated_coords[:,0]
            y_i = self.rotated_coords[:,1]-self.parent.rotated_coords[:,1]
            z_i = self.rotated_coords[:,2]-self.parent.rotated_coords[:,2]
            l_i = (x_i**2 + y_i**2 + z_i**2)**0.5 + 0.00001
            
            self._alpha = (np.arccos(x_i/l_i))
            self._beta =  (np.arccos(y_i/l_i))
            self._gamma = (np.arccos(z_i/l_i))

            rel_a = np.zeros(self.rotated_coords.shape[0])
            special_case = [5,9,13,17]
            if (self.id-mediator) in special_case:
                for i in range(self.rotated_coords.shape[0]):
                    rel_a[i] = angle_rad((self.rotated_coords[i]-self[body_parent].rotated_coords[i]),(self[body_parent].parent.rotated_coords[i]-self[body_parent].rotated_coords[i]))
            else:
                for i in range(self.rotated_coords.shape[0]):
                    rel_a[i] = angle_rad((self.rotated_coords[i]-self.parent.rotated_coords[i]),(self.parent.parent.rotated_coords[i]-self.parent.rotated_coords[i]))
            self.rel_a = rel_a
            self.relative_color = relative_angle_to_RGB_uint8(self.rel_a)

            self.color = np.vstack((angle_to_uint8(self._alpha),angle_to_uint8(self._beta),angle_to_uint8(self._gamma))).T
            self.signal = np.vstack((self._alpha,self._beta,self._gamma)).T
        return

    def _f_transform_coords(self):
        #if hasattr(self,'normed_coords'):
        #print('Tr',self.id)
        self._get_rotated_coords(2)
        self._get_face_rotated_coords()

        if self.id!=1001 and self.id!=1006:

            c =   self._get_root().relaxed_face[str(self.id)][0]
            x_i = self.face_rotated_coords[:,0] - c[0]
            y_i = self.face_rotated_coords[:,1] - c[1]
            z_i = self.face_rotated_coords[:,2] - c[2]
            l_i = (c[0]**2 + c[1]**2 + c[2]**2)**0.5 + 0.000001

            _alpha = (np.arccos(x_i/l_i))
            _beta =  (np.arccos(y_i/l_i))
            _gamma = (np.arccos(z_i/l_i))

            self._alpha = ((_alpha-(np.pi*0.5))*4) + (np.pi*0.5)
            self._beta = ((_beta -(np.pi*0.5))*4 ) + (np.pi*0.5)
            self._gamma = ((_gamma-(np.pi*0.5))*4 ) + (np.pi*0.5)
        else:
            x_i = self.rotated_coords[:,0]-self.parent.rotated_coords[:,0]
            y_i = self.rotated_coords[:,1]-self.parent.rotated_coords[:,1]
            z_i = self.rotated_coords[:,2]-self.parent.rotated_coords[:,2]

            l_i = (x_i**2 + y_i**2 + z_i**2)**0.5 + 0.000001
             
            self._alpha = (np.arccos(x_i/l_i))
            self._beta =  (np.arccos(y_i/l_i))
            self._gamma = (np.arccos(z_i/l_i))
            
        self.color = np.vstack((angle_to_uint8(self._alpha),angle_to_uint8(self._beta),angle_to_uint8(self._gamma))).T
        self.signal = np.vstack((self._alpha,self._beta,self._gamma)).T
        return

    def _print_tree(self):
        if len(self.children)!=0:
            print(self.id, list([c.id for c in self.children]), self.dist_to_parent)
            for i in self.children: 
                i._print_tree()
        else:
            print(self.id, list([]), self.dist_to_parent)
            return
            
    def _draw_tree(self, image, fr, rotated, joint_list=[], text='', add_face=False):
        if self.is_root() == False: 
            self._get_root()._draw_tree(image,fr,rotated,joint_list)
        else:

            if joint_list == []:
                joint_list = list([j.id for j in self])

            for j in joint_list:
                    d_only = False
                    #d_color = (255,255,255)
                    pr_text = True
                    d_radius = 3
                    l_color= (128,128,128)
                    f_color = (255,255,255)
                    #f_color = (0,0,0)

                    #l_color = (0,0,0)
                    #f_color = (0,0,0)
                    d_color = (0,0,0)
                    #ROOT
                    if j==1:
                        d_only = True
                        d_color = (0,0,255)
                        pr_text = False

                    #HANDS
                    if 400<j<811:
                        pr_text = False
                        d_radius=2
                        d_color = (255,255,255)
                    #FACE
                    if j>999:
                        pr_text = False
                        d_only = True
                        d_radius=2
                        d_color = (255,255,255)
                    #if j!=999: #EXCEPTION

                    image = self._get_joint(j)._draw_joint(image,fr,rotated,dot_only=d_only,dot_radius=d_radius,dot_color=d_color,print_text=pr_text,line_color=l_color,font_color=f_color)

        if text == '':
            text = ['Frame:', fr]
        text_color = (0,0,0)#(255,255,255)
        cv2.putText(image,str(text), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color)

        if add_face:
            img_face = np.zeros_like(image)
            #img_face.fill(255)
            img_face = self._draw_face(img_face,fr)
            img_face = cv2.resize(img_face,(int(image.shape[0]*0.35),int(image.shape[1]*0.35)))
            #print(img_face.shape)
            image[:img_face.shape[0],-img_face.shape[1]:] = img_face[:,:]
        return image
              
    def _draw_joint(self,image,fr,rotated,dot_only=False,print_text=False,dot_radius=2,dot_color=(255,0,0),line_thickness=2,line_color=(0,0,0), font_size=0.4,font_color=(0,0,0)):
        w, h = image.shape[0], image.shape[1]
        if hasattr(self,'color') & rotated:
            dot_color = self.color[fr]
            line_color = dot_color
            if self.level>=3:
                if hasattr(self,'relative_color') & rotated:
                    line_color = self.relative_color[fr] 

        if dot_only== False:
            if rotated:
                start_point = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.parent.rotated_coords[fr][:2]))))
                end_point   = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.rotated_coords[fr][:2]))))
            else:
                start_point = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.parent.normed_coords[fr][:2]*2))))
                end_point   = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.normed_coords[fr][:2]*2))))
        else:
            if rotated:
                end_point   = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.rotated_coords[fr][:2]))))
            else:
                end_point   = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.normed_coords[fr][:2]*2))))

        try:
            if dot_only == False:

                thickness = line_thickness
                #row,col = draw.line(start_point[1],start_point[0],end_point[1],end_point[0])
                #image[row,col,:]= np.array(color,dtype=np.uint8)
                image = cv2.line(image, start_point, end_point, line_color, thickness)
            try:
                #row, col = draw.disk(end_point, dot_radius)
                #image[col,row,:]= np.array(dot_color,dtype=np.uint8)
                #cv2.circle(image,end_point,dot_radius,dot_color,-1,8)
                dot = np.zeros((dot_radius*2,dot_radius*2,3),dtype=np.uint8)
                dot[:,:] = np.array(dot_color,dtype=np.uint8)

                image[end_point[1]-dot_radius:end_point[1]+dot_radius,end_point[0]-dot_radius:end_point[0]+dot_radius] = dot

            except:
                pass
                
            if print_text:
                if rotated:
                    text = str(tuple(map(int,(self.rotated_coords[fr][:3]))))
                    if self.level>=3:
                        if hasattr(self,'relative_color') & rotated:
                            #line_color = self.relative_color[fr] 
                            text = str(self.id)+" "+str(round(np.degrees(self.rel_a[fr])))
                    else:
                        text = str(self.id)#+" "+str(tuple(map(int,(np.array(self.color[fr],dtype=int)))))
                        #text = ''
                else:
                    text = str(tuple(map(int,(self.normed_coords[fr][:3]))))
                    text = str(self.id)+" "+str(tuple(map(int,(np.array(self.color[fr],dtype=int)))))

                cv2.putText(image,text,(end_point[0]+10,end_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color)
        except:
            pass
        return image
    
    def _draw_face(self, image, fr, joint_list=[], text=''):
        #print("DF")
        if self.id != 999: 
            self[999]._draw_face(image,fr,joint_list,text)
        else:

            if joint_list == []:
                joint_list = list([j.id for j in self.children])

            for j in joint_list:
                    #d_color = (255,255,255)
                    pr_text = False
                    d_radius = 5
                    #f_color = (255,255,255)

                    #if j!=999: #EXCEPTION
                    #print(j)
                    #l_color = (0,0,0)
                    f_color = (0,0,0)
                    d_color = (0,0,0)
                    image = self._get_joint(j)._draw_face_dot(image,fr,dot_radius=d_radius,dot_color=d_color,print_text=pr_text,font_color=f_color)

        if text != '':
            text = text
        f_color = (255,255,255)
        cv2.putText(image,str(text), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, f_color)

        return image

    def _draw_face_dot(self,image,fr,print_text=False,dot_radius=2,dot_color=(255,0,0), font_size=0.5,font_color=(0,0,0)):
        #print(self.id)
        w, h = image.shape[0], image.shape[1]
        if hasattr(self,'color'):
            dot_color = self.color[fr]

        end_point = tuple(map(int,(np.array([w/2,h/2],dtype=int) + (self.face_rotated_coords[fr][:2]*2.5))))
        try:
            try:
                dot = np.zeros((dot_radius*2,dot_radius*2,3),dtype=np.uint8)
                dot[:,:] = np.array(dot_color,dtype=np.uint8)
                image[end_point[1]-dot_radius:end_point[1]+dot_radius,end_point[0]-dot_radius:end_point[0]+dot_radius] = dot
            except:
                pass

            if print_text:
                #text = str(tuple(map(int,(self.face_rotated_coords[fr][:3]))))
                text = str(self.id - 1000)
                cv2.putText(image,text,(end_point[0],end_point[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color)
        except:
            pass
        return image

    def posegram(self, human=False, signal= False):
        if human:
            #angles in radians converted to uint8 values and RGB combined for each joint -> (rel_joints + joints)x(frames) uint8 RGB
            return self._posegram_human()
        else:
            if signal:
                #raw angles in radians float values -> (rel_joints + (joints x 3)) x(frames) float binary
                return self._posegram_machine_signal()
            else:
                #angles in radians converted to uint8 values -> (rel_joints + (joints x 3)) x (frames) uint8 RGB
                return self._posegram_machine()

    def _posegram_machine(self):
        joints = list(int(k.id) for k in self._get_root())
        rel_joints =list(int(k.id) for k in self._get_root() if k.level>=3 )

        img = np.zeros((len(joints),self.color.shape[0],3),dtype=np.uint8)

        i = 0
        for j in joints:
            img[i,:] =  self[j].color
            #print(i,j)
            i+=1

        r = img[:,:,0]
        r = r.astype(np.uint8)
        g = img[:,:,1]
        g = g.astype(np.uint8)
        b = img[:,:,2]
        b = b.astype(np.uint8)
        img  = np.concatenate((r,g,b),axis=0)

        img_rel = np.zeros((len(rel_joints),self.color.shape[0]),dtype=np.uint8)

        r = 0
        for j in rel_joints:
            img_rel[r,:] = angle_to_uint8(self[j].rel_a)
            r+=1  

        img  = np.concatenate((img_rel,img),axis=0)

        return img

    def _posegram_machine_signal(self):
        joints = list(int(k.id) for k in self._get_root())
        rel_joints =list(int(k.id) for k in self._get_root() if k.level>=3)

        data = np.zeros((len(joints),self.signal.shape[0],3))
        i = 0
        for j in joints:
            data[i,:] = self[j].signal
            i+=1

        x = data[:,:,0]
        y = data[:,:,1]
        z = data[:,:,2]
        data  = np.concatenate((x,y,z),axis=0)

        data_rel = np.zeros((len(rel_joints),self.signal.shape[0]))

        r = 0
        for j in rel_joints:
            data_rel[r,:] = self[j].rel_a
            r+=1  

        data  = np.concatenate((data_rel,data),axis=0)

        return data 

    def _posegram_human(self):
        joints = list(int(k.id) for k in self._get_root())
        rel_joints =list(int(k.id) for k in self._get_root() if k.level>=3 )
        
        img = np.zeros((len(joints),self.color.shape[0],3),dtype=np.uint8)

        i = 0
        for j in joints:
            img[i,:] =  self[j].color
            i+=1
        
        img_rel = np.zeros((len(rel_joints),self.color.shape[0],3),dtype=np.uint8)

        r = 0
        for j in rel_joints:
            img_rel[r,:] = relative_angle_to_RGB_uint8(self[j].rel_a)
            r+=1  

        img  = np.concatenate((img_rel,img),axis=0)

        return img

    def _basic_normalization_mov_data(self, to_uint8=False):
        joints = list(int(k.id) for k in self._get_root())
        data = np.zeros((len(joints),self.joint_datum.shape[0],3))
        i = 0
        for j in joints:
            data[i,:] = self[j].basic_normalization_coords
            i+=1

        x = data[:,:,0]
        y = data[:,:,1]
        z = data[:,:,2]
        data  = np.concatenate((x,y,z),axis=0)
        if to_uint8:
            data = angle_to_uint8(data)
            data = data.astype(np.uint8)    
        return data 

    def _posegram_grayscale_uint8(self):
        return self._posegram_machine()

    def _posegram_color_uint8(self):
        return self._posegram_human()

    def _posegram_rad_float(self):
        return self._posegram_machine_signal()

#Movement is a wraper for Joint_Tree with extra init functions
class Movement(Joint_Tree):
    def __init__(self, movement=None, verbose=True,  holistic = None):
        if type(movement) == type('path'):
            super().__init__()
            if movement.find('.joblib')!=-1:
                mov = joblib.load(movement)
            elif movement.find('.mp4')!=-1 or movement.find('.MOV')!=-1 or movement.find('.MP4')!=-1 or movement.find('.mov')!=-1:
                mov = self.movement_from_mediapipe(movement,verbose,holistic)
            self.process(mov)
        elif type(movement) == type({}):
            super().__init__()
            self.process(movement)
        elif type(movement) == type(np.array([])):
            super().__init__()
            if type(abs(movement.item(0))) ==type(int(0)):
                self.from_gram_process(movement)
            if type(abs(movement.item(0))) ==type(float(0)):
                self.from_gram_process(signal_to_uint(np.absolute(movement)))
        else:
            super().__init__()

    def movement_from_mediapipe(self, video_path, verbose = True, holistic = None):
        def process_frame_with_mp(holistic,in_frame):
            in_frame.flags.writeable = False
            results = holistic.process(in_frame)
            in_frame.flags.writeable = True
            return results

        mov_dict = {}
        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        else:
            if verbose:
                print("Processing with MediaPipe video at:",video_path)
        
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps
        #MP_Face = np.zeros((frames_total,468,4))
        MP_Face = np.zeros((frames_total,478,4)) #please set refine fafe landmarks True
        MP_Pose = np.zeros((frames_total,33,7))
        MP_RHand = np.zeros((frames_total,21,4))
        MP_LHand = np.zeros((frames_total,21,4))
        fr = 0
        if holistic:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    new_result  = process_frame_with_mp(holistic,frame)
                    if new_result:
                        if new_result.face_landmarks:
                            MP_Face[fr] = mp_frame_coords(new_result.face_landmarks,frame_height,frame_width)
                        if new_result.pose_landmarks:
                            MP_Pose[fr] = mp_frame_coords(new_result.pose_landmarks,frame_height,frame_width,new_result.pose_world_landmarks)
                        if new_result.right_hand_landmarks:
                            MP_RHand[fr] = mp_frame_coords(new_result.right_hand_landmarks,frame_height,frame_width)
                        if new_result.left_hand_landmarks:
                            MP_LHand[fr] = mp_frame_coords(new_result.left_hand_landmarks,frame_height,frame_width)
                    fr+=1
                    if ((fr/frames_total)*100)%10==0 and verbose:
                        print('Progress ',(int((fr/frames_total)*100)), "%",end='\r')
                else: 
                    break
        else:
            mp_holistic = mp.solutions.holistic
            with mp_holistic.Holistic(
                smooth_landmarks=True,
                model_complexity=2,
                min_detection_confidence=0.1,
                refine_face_landmarks=True,
                min_tracking_confidence=0.1) as holistic:
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        new_result  = process_frame_with_mp(holistic,frame)
                        if new_result:
                            if new_result.face_landmarks:
                                MP_Face[fr] = mp_frame_coords(new_result.face_landmarks,frame_height,frame_width)
                            if new_result.pose_landmarks:
                                MP_Pose[fr] = mp_frame_coords(new_result.pose_landmarks,frame_height,frame_width,new_result.pose_world_landmarks)
                            if new_result.right_hand_landmarks:
                                MP_RHand[fr] = mp_frame_coords(new_result.right_hand_landmarks,frame_height,frame_width)
                            if new_result.left_hand_landmarks:
                                MP_LHand[fr] = mp_frame_coords(new_result.left_hand_landmarks,frame_height,frame_width)
                        fr+=1
                        if ((fr/frames_total)*100)%10==0 and verbose:
                            print('Progress ',(int((fr/frames_total)*100)), "%",end='\r')
                    else: 
                        break
                del(mp_holistic)
            #holistic.__exit__()

        cap.release()
        cv2.destroyAllWindows()   
        #print("released")
        mov_dict = {'MP_Pose':MP_Pose,
                    'MP_Face':MP_Face,
                    'MP_RHand':MP_RHand,
                    'MP_LHand':MP_LHand}

        mov_dict['Meta'] = {'Video Path':video_path,
                            'H':frame_height,
                            'W':frame_width,
                            'FPS':fps,
                            'Frames':frames_total}
        out_name_i = video_path.rfind('.')
        #print(video_path[:out_name_i]+'.joblib')
        #joblib.dump(mov_dict,video_path[:out_name_i]+'.joblib')
        del(cap,MP_Pose,MP_Face,MP_LHand,MP_RHand)
        return mov_dict

    def from_gram_process(self,gram): 
        def get_face_rotation_from_rel(inp_x,inp_1,tar_1,inp_6,tar_6):
            i_x = np.ones(4)
            i_x[:3] = inp_x
            i_1 = np.ones(4)
            i_1[:3] = inp_1
            i_6 = np.ones(4)
            i_6[:3] = inp_6
            i_t1 = np.ones(4)
            i_t1[:3] = tar_1
            i_t6 = np.ones(4)
            i_t6[:3] = tar_6

            rot_mat = get_face_rotation_mat(i_1,i_t1,i_6,i_t6)
            rotated_c = rotate(i_x,rot_mat)
            scale_coeff = abs((np.linalg.norm(tar_1))/(np.linalg.norm(inp_1)+0.000001))
            rotated_coords = scale(rotated_c,scale_coeff)
            return rotated_coords[:3]

        joints = list(k for k in self)
        rel_joints =list(int(k.id) for k in self if k.level>=3 )
        r = len(rel_joints)
        
        if len(gram.shape)==3: #from human            
            gram = np.transpose(gram[44:,:,:],(2,0,1))
            gram = gram.reshape((gram.shape[0]*gram.shape[1],gram.shape[2]))

        if gram.shape[0]==411:
            r = 0 
        print(gram.shape)
        jn = len(joints)
        for i in range(len(joints)):
            j = joints[i]
            _alpha = np.zeros((gram.shape[1]))
            _beta = np.zeros((gram.shape[1]))
            _gamma = np.zeros((gram.shape[1]))
            
            rotated_coords = np.zeros((gram.shape[1],3))
            if j.id!=1:
                if j.id>999:
                    face_rotated_coords = np.zeros((gram.shape[1],3))
                    c = j._get_root().relaxed_face[str(j.id)][0]
                    if j.id!=1001 and j.id!=1006:
                        _alpha = uint8_to_angle(gram[r+i,:])
                        _beta  = uint8_to_angle(gram[r+jn+i,:])
                        _gamma = uint8_to_angle(gram[r+(jn*2)+i,:])

                        _alpha = ((_alpha - (np.pi*0.5))/4) +(np.pi*0.5) 
                        _beta  = ((_beta - (np.pi*0.5))/4) +(np.pi*0.5)
                        _gamma = ((_gamma - (np.pi*0.5))/4) +(np.pi*0.5)

                        l_i = (c[0]**2 + c[1]**2 + c[2]**2)**0.5 + 0.000001

                        x_i = np.cos(_alpha)*l_i
                        y_i = np.cos(_beta)*l_i
                        z_i = np.cos(_gamma)*l_i
                        
                        face_rotated_coords[:,0] = c[0] + x_i
                        face_rotated_coords[:,1] = c[1] + y_i
                        face_rotated_coords[:,2] = c[2] + z_i
                    else:
                        face_rotated_coords[:] = c
                    j.face_rotated_coords = face_rotated_coords

                
                if j.id>999 and j.id!=1001 and j.id!=1006:
                    for fr in range(rotated_coords.shape[0]):
                        head_origin_1 = self[1001].rotated_coords[fr] - self[1001].parent.rotated_coords[fr]
                        head_origin_6 = self[1006].rotated_coords[fr] - self[1006].parent.rotated_coords[fr]
                        
                        rotated_c = get_face_rotation_from_rel(face_rotated_coords[fr],
                                                                self[1001].face_rotated_coords[fr],
                                                                head_origin_1,
                                                                self[1006].face_rotated_coords[fr],
                                                                head_origin_6)
                        rotated_coords[fr] = j.parent.rotated_coords[fr] + rotated_c
                else:
                    _alpha = uint8_to_angle(gram[r+i,:])
                    _beta  = uint8_to_angle(gram[r+jn+i,:])
                    _gamma = uint8_to_angle(gram[r+(jn*2)+i,:])
                    
                    l_i = j.dist_to_parent * 200
                    x_i = np.cos(_alpha)*l_i
                    y_i = np.cos(_beta)*l_i
                    z_i = np.cos(_gamma)*l_i
                    
                    rotated_coords[:,0] = j.parent.rotated_coords[:,0] + x_i
                    rotated_coords[:,1] = j.parent.rotated_coords[:,1] + y_i
                    rotated_coords[:,2] = j.parent.rotated_coords[:,2] + z_i

            j.rotated_coords = rotated_coords

            ##RECREATING THE SIGNAL AND COLOR CODING
            if j.dist_to_parent!=0:

                if j.id>999 and j.id!=1001 and j.id!=1006:
                    c =   j._get_root().relaxed_face[str(j.id)][0]

                    x_i = j.face_rotated_coords[:,0] - c[0]
                    y_i = j.face_rotated_coords[:,1] - c[1]
                    z_i = j.face_rotated_coords[:,2] - c[2]
                    l_i = (c[0]**2 + c[1]**2 + c[2]**2)**0.5 + 0.000001

                    _alpha = (np.arccos(x_i/l_i))
                    _beta =  (np.arccos(y_i/l_i))
                    _gamma = (np.arccos(z_i/l_i))

                    _alpha = ((_alpha-(np.pi*0.5))*4) + (np.pi*0.5)
                    _beta = ((_beta -(np.pi*0.5))*4 ) + (np.pi*0.5)
                    _gamma = ((_gamma-(np.pi*0.5))*4 ) + (np.pi*0.5)

                else:
                    #print(j.id)
                    x_i = j.rotated_coords[:,0]-j.parent.rotated_coords[:,0]
                    y_i = j.rotated_coords[:,1]-j.parent.rotated_coords[:,1]
                    z_i = j.rotated_coords[:,2]-j.parent.rotated_coords[:,2]
                    l_i = (x_i**2 + y_i**2 + z_i**2)**0.5 + 0.000001
                    
                    _alpha = (np.arccos(x_i/l_i))
                    _beta =  (np.arccos(y_i/l_i))
                    _gamma = (np.arccos(z_i/l_i))

            j._alpha = _alpha
            j._beta = _beta
            j._gamma = _gamma


            if j.level >= 3:
                rel_a = np.zeros(j.rotated_coords.shape[0])
                special_case_7 = [705,709,713,717]
                special_case_4 = [405,409,413,417]
                if j.id in special_case_7:
                    for i in range(j.rotated_coords.shape[0]):
                        rel_a[i] = angle_rad((j.rotated_coords[i]-j[7].rotated_coords[i]),(j[7].parent.rotated_coords[i]-j[7].rotated_coords[i]))
                elif j.id in special_case_4:
                    for i in range(j.rotated_coords.shape[0]):
                        rel_a[i] = angle_rad((j.rotated_coords[i]-j[4].rotated_coords[i]),(j[4].parent.rotated_coords[i]-j[4].rotated_coords[i]))
                else:
                    for i in range(j.rotated_coords.shape[0]):
                        rel_a[i] = angle_rad((j.rotated_coords[i]-j.parent.rotated_coords[i]),(j.parent.parent.rotated_coords[i]-j.parent.rotated_coords[i]))
                j.rel_a = rel_a
                j.relative_color = relative_angle_to_RGB_uint8(j.rel_a)

            j.color = np.vstack((angle_to_uint8(j._alpha),angle_to_uint8(j._beta),angle_to_uint8(j._gamma))).T    
            j.signal = np.vstack((j._alpha,j._beta,j._gamma)).T
        return

    def make_a_video(self, out_path, fps = 60):
        X_DIMENSION = 600
        Y_DIMENSION = 600
        if hasattr(self,'fps'):
            fps = self.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (X_DIMENSION, Y_DIMENSION))
        total = self.rotated_coords.shape[0]

        for fr in range(total):

            black_image = np.zeros((X_DIMENSION, Y_DIMENSION,3),dtype=np.uint8)
            black_image = self._draw_tree(black_image,fr,True,[],add_face=True)
            black_image = cv2.cvtColor(black_image.astype(np.uint8),cv2.COLOR_BGR2RGB)
            out.write(black_image)

            #if ((fr/total)*100)%10==0:
            #    print('Progress ',(int((fr/total)*100)), "%")
        out.release()

### FUNCTIONS ###

def distance_two_points(p1, p2):
    if all(p1)==False or all(p2)==False:
        return 0
    dx = abs(p1[0]-p2[0])
    dy = abs(p1[1]-p2[1])
    dist = ( dx**2 + dy**2)**0.5
    return dist

def x_distance_two_points(p1, p2):
    if all(p1)==False or all(p2)==False:
        return 0
    dx = (p1[0]-p2[0])
    return dx

def y_distance_two_points(p1, p2):
    if all(p1)==False or all(p2)==False:
        return 0

    dy = (p1[1]-p2[1])
    return dy

def a_distance_two_points_3d(a1,a2):
    dx = abs(a1[:,0] - a2[:,0])
    dy = abs(a1[:,1] - a2[:,1])
    dz = abs(a1[:,2] - a2[:,2])
    dist = ( dx**2 + dy**2 + dz**2 + 0.00001)**0.5
    return dist

def a_distance_two_points_3d_z_low(a1,a2):
    dx = abs(a1[:,0] - a2[:,0])
    dy = abs(a1[:,1] - a2[:,1])
    dz = abs(a1[:,2] - a2[:,2]) / 2
    dist = ( dx**2 + dy**2 + dz**2 + 0.00001)**0.5
    return dist

def get_rotation_mat(inp, tar):
     
    rot_y, yr = y_rot_m(inp,tar) 
    #the X rotation is usially not needed or have to be worked on, specifying the target rotation
    #rot_x, xr = x_rot_m(rot_y,tar)
    rot_z, zr = z_rot_m(rot_y,tar)

    return np.dot(yr,zr)

def get_face_rotation_mat(inp, tar, inp1006, tar1006):
    rot_x, xr = x_rot_m(inp,tar)
    rot_y, yr = y_rot_m(rot_x,tar) 
    #tar1006 = np.array([0,-1,0,1])
    inp1006= np.dot(inp1006,np.dot(xr,yr))
    rot_z, zr = z_rot_m(inp1006,tar1006)
    return np.dot(np.dot(xr,yr),zr)

def angle_rad(v1, v2):
    up = np.dot(v1, v2)
    down = (np.linalg.norm(v1)*np.linalg.norm(v2))  + 0.0000000001

    angle = np.arccos(up/down)        
    return angle

def x_rot_m(inp,tar):
    #print(inp,tar)
    #print(inp[1:3],tar[1:3]+np.array([tar[0]/abs(tar[0]),0]))
    #x = angle_rad(inp[1:3],tar[1:3]+np.array([tar[0]/abs(tar[0]),0]))
    x = angle_rad(inp[1:3],tar[1:3])

    xr = [[1,0,0,0],
         [0,np.cos(x),-np.sin(x),0],
         [0,np.sin(x),np.cos(x),0],
         [0,0,0,1]]

    rot_x = np.dot(inp,xr)

    #x2 = angle_rad(rot_x[1:3],tar[1:3]+np.array([tar[0]/abs(tar[0]),0]))
    x2 = angle_rad(rot_x[1:3],tar[1:3])

    if x<x2:

        x = -x 
        xr = [[1,0,0,0],
         [0,np.cos(x),-np.sin(x),0],
         [0,np.sin(x),np.cos(x),0],
         [0,0,0,1]]
        rot_x = np.dot(inp,xr)

        #x2 = ang(rot_x[1:3],tar[1:3]+np.array([tar[0]/abs(tar[0]),0]))

    return rot_x, xr

def y_rot_m(inp,tar):
    y = angle_rad([inp[0],inp[2]],[tar[0],tar[2]])

    yr = [[np.cos(y),0,np.sin(y),0],
                [0,1,0,0],
                [-np.sin(y),0,np.cos(y),0],
                [0,0,0,1]]
    
    rot_y = np.dot(inp,yr)

    y2 = angle_rad([rot_y[0],rot_y[2]],[tar[0],tar[2]])

    if y<y2:
        y = -y 
        yr = [[np.cos(y),0,np.sin(y),0],
                [0,1,0,0],
                [-np.sin(y),0,np.cos(y),0],
                [0,0,0,1]]
        rot_y = np.dot(inp,yr)
        #y2 = -angle_rad([inp[0],inp[2]],[tar[0],tar[2]])
    return rot_y, yr

def z_rot_m(inp,tar):
    z = angle_rad(inp[:2],tar[:2])

    zr = [[np.cos(z),-np.sin(z),0,0],
            [np.sin(z) ,np.cos(z)  ,0,0],
            [0,0,1,0],
            [0,0,0,1]]
    
    rot_z = np.dot(inp,zr)

    z2 = angle_rad(rot_z[:2],tar[:2])
    z3= 0
    if z<z2:

        z = -z 
        zr = [[np.cos(z),-np.sin(z),0,0],
        [np.sin(z) ,np.cos(z)  ,0,0],
        [0,0,1,0],
        [0,0,0,1]]
        rot_z = np.dot(inp,zr)
        z3 = angle_rad(rot_z[:2],tar[:2])

    return rot_z, zr

def rotate(inp, rot):
    out = np.dot(inp,rot)
    return out

def scale(inp,scale_coeff):
    scale_mat =[[scale_coeff,0,0,0],
                [0,scale_coeff,0,0],
                [0,0,scale_coeff,0],
                [0,0,0,1]]
    out = np.dot(inp,scale_mat)

    return out[:3]

def mp_frame_coords(landmarks, H, W, extra_pose_lm=None):
    if extra_pose_lm:
        coord = np.zeros((len(landmarks.landmark),7))
        for i in range(len(landmarks.landmark)):
            coord[i][0] = (landmarks.landmark[i].x * W)
            coord[i][1] = (landmarks.landmark[i].y * H)
            coord[i][2] = (-landmarks.landmark[i].z * W)
            coord[i][3] = (landmarks.landmark[i].visibility)
            coord[i][4] = extra_pose_lm.landmark[i].x * 100
            coord[i][5] = extra_pose_lm.landmark[i].y * 100
            coord[i][6] = -extra_pose_lm.landmark[i].z * 100
    else:
        coord = np.zeros((len(landmarks.landmark),4))
        for i in range(len(landmarks.landmark)):
            coord[i][0] = (landmarks.landmark[i].x * W)
            coord[i][1] = (landmarks.landmark[i].y * H)
            coord[i][2] = (-landmarks.landmark[i].z * W)
            coord[i][3] = (landmarks.landmark[i].visibility)
    return coord

def angle_to_uint8(angle):
    x = (angle * 128)/np.pi +128 
    x = np.where(x < 255, x, np.array([255 for i in range(x.shape[0])]))
    return x

def uint8_to_angle(int8):
    angle = (int8-128)/128 * np.pi
    return angle

def relative_angle_to_RGB_uint8(angle):

    R,G,B = np.zeros_like(angle),np.zeros_like(angle),np.zeros_like(angle)
    #b for 0, r for 90, g for 180
    for i in range(angle.shape[0]):
        if angle[i] > (np.pi*0.5):

            R[i] = abs(np.sin(angle[i])*255)
            G[i] = abs(np.cos(angle[i])*255)
            B[i] = 0
        else:

            R[i] = abs(np.sin(angle[i])*255)
            G[i] = 0
            B[i] = abs(np.cos(angle[i])*255)
            
    return np.vstack((R,G,B)).T

def signal_to_uint(signal_gram):
    rgb_gram = np.zeros_like(signal_gram)
    for s in range(signal_gram.shape[0]):
        rgb_gram[s] = angle_to_uint8(signal_gram[s])
    rgb_gram = rgb_gram.astype(np.uint8)
    return rgb_gram

def uint_to_signal(rgb_gram):
    signal_gram = np.zeros_like(rgb_gram)
    for s in range(rgb_gram.shape[0]):
        signal_gram[s] = uint8_to_angle(rgb_gram[s])
    return signal_gram

def interpolate_zeros(inp):       
    xp = np.nonzero(inp)[0]
    all_xp = np.arange(len(inp))
    x = np.array([a for a in all_xp if a not in xp]).astype(int)
    fp = inp[np.nonzero(inp)[0].astype(int)]
    iterp = np.interp(x,xp,fp)
    inp[x] = iterp
    return inp

def make_a_video(m, out_path):
    X_DIMENSION = 600
    Y_DIMENSION = 600
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 60, (X_DIMENSION, Y_DIMENSION))
    total = m.rotated_coords.shape[0]
    
    for fr in range(total):

        black_image = np.zeros((X_DIMENSION, Y_DIMENSION,3),dtype=np.uint8)
        black_image = m._draw_tree(black_image,fr,True,[],add_face=True)
        black_image = cv2.cvtColor(black_image.astype(np.uint8),cv2.COLOR_BGR2RGB)
        out.write(black_image)

        #if ((fr/total)*100)%10==0:
        #    print('Progress ',(int((fr/total)*100)), "%")
    out.release()
    
def smooth_out(inp, win_size=10, times=1, weights=[]):
    out = inp
    w_ = weights
    for i in range(times):
        win = window(out,win_size)
        if len(w_)!=len(inp):
            out  = list(np.average(w) for w in win )
        else:
            w_win = window(w_,win_size)
            out  = list(np.average(win[i],weights=w_win[i]) for i in range(len(win)))
            
    return out

def resize_sample(gram, arg):
    print(type(gram), type(arg))
    if type(arg)==type(0.1):
        return resize_sample_as_factor(gram,arg)
    if type(arg)==type(gram):
        return resize_sample_as_target(gram,arg)

def resize_sample_as_factor(gram, factor):
    gram = np.array(gram)
    new_shape = list(gram.shape)
    new_shape[1] = round(new_shape[1]*factor)
    res_gram = np.resize(gram,tuple(new_shape))
    for i in range(gram.shape[0]):
        res_gram[i] = np.interp(np.arange(0, round(gram[i].shape[0]*factor)), np.arange(0, gram[i].shape[0])*factor, gram[i])
    return res_gram

def resize_sample_as_target(gram, target_gram):
    gram = np.array(gram)
    
    factor = target_gram.shape[1]/gram.shape[1]
    res_gram = np.resize(gram,(gram.shape[0],target_gram.shape[1]))
    for i in range(gram.shape[0]):
        res_gram[i] = np.interp(np.arange(0, target_gram.shape[1]), np.arange(0, gram[i].shape[0])*factor, gram[i])
    return res_gram

def trim_posegram(pg):
    #remove face and relative joint data from the gram
    if pg.shape[0]==455:
        pg = pg[44:]
    pg = np.concatenate((pg[:53],pg[53+84:84+2*53],pg[2*84+2*53:2*84+3*53]),axis=0)
    return pg

def add_to_trimmed(pg, ex):
    #add face and relative joint data from the example gram
    #print(ex.shape)
    exp = np.copy(ex)
    #print(exp.shape)
    if exp.shape[0]==455:
        r = 44
    else:
        r = 0

    if exp.shape[1] != pg.shape[1]:
        exp.resize((exp.shape[0], pg.shape[1]))
    #print(exp.shape)
    exp[r:r+53] = pg[:53]#adding x
    exp[r+53+84:r+53*2+84] = pg[53:53*2]#adding y
    exp[r+53*2+84*2:r+53*3+84*2] = pg[53*2:53*3]#adding z

    return exp

def t2p_2_gram(t2p, ex):
    t2p = ((t2p*128)+128).astype(np.uint8)
    t2p = add_to_trimmed(t2p, ex)
    return t2p

def get_files(input_dir_path, ext = '.mov'):
    files = []
    for root, dirs, fil in os.walk(input_dir_path):
        for f in fil:
            if str(f).endswith(ext) and str(f)[:2]!='._':
                #print(os.path.join(root, file))
                #print(file)
                files.append([str(f),str(f)[:-len(ext)],os.path.join(root, f)])
    return files

def get_dirs(input_dir_path):
    folders = []
    for root, dirs, files in os.walk(input_dir_path):
        for d in dirs:
            folders.append([str(d),os.path.join(root, d)])
    return folders

def window(seq, win_l = 15):
    win_l = win_l//2
    w = []
    for index in range(len(seq)):
        if index+win_l >len(seq):
            window_ = [seq[-1] for i in range(0,((win_l*2)))]
            window_[:-(win_l-(len(seq)-index))] = seq[index-win_l:]
        else:    
            if index < win_l:
                window_ = [seq[1] for i in range(0,((win_l*2)))]
                window_[win_l-index:] = seq[:(index+win_l)]
            else:
                window_ = [0 for i in range(0,((win_l*2)))]
                window_ = seq[index-win_l:index+win_l]
        w.append(window_)
    return  w
