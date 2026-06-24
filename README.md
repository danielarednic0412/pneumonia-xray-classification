Detectarea Pneumoniei din Radiografii cu Deep Learning

Acesta e proiectul meu de licență. Am vrut să văd dacă un model de AI poate clasifica radiografii toracice ca fiind normale sau cu pneumonie, și cât de bine se descurcă arhitecturi diferite pe același dataset.
Am implementat totul în două medii — MATLAB și Python pe Google Colab — ca să pot compara nu doar arhitecturile, ci și framework-urile.

Ce am obținut
Mediu Model Acuratețe MATLABResNet-18 91.51%  PythonResNet-18   90.54%   PythonResNet-50  87.50%
Surprinzător (sau poate nu), ResNet-18 a bătut ResNet-50 pe acest dataset. Cel mai probabil pentru că datasetul nu e suficient de mare ca să justifice o rețea mai complexă — ResNet-50 a început să memoreze în loc să generalizeze.

Dataset
Chest X-Ray Images — Kaggle
~5.800 de radiografii împărțite în train/val/test, două clase: NORMAL și PNEUMONIA.

Ce tehnologii am folosit:
MATLAB Deep Learning Toolbox
Python, TensorFlow/Keras, PyTorch
Google Colab
Transfer learning cu ResNet-18 și ResNet-50 pre-antrenate pe ImageNet
Grad-CAM pentru hărți de activare — să văd ce zone din radiografie influențează decizia modelului
Dropout, BatchNormalization freeze, EarlyStopping, ReduceLROnPlateau


Cum am antrenat
N-am antrenat de la zero — am luat rețele deja pre-antrenate pe ImageNet și le-am adaptat pentru problema asta (transfer learning).
Procesul a fost în două faze:

Înghețăm rețeaua de bază și antrenăm doar straturile noi adăugate de mine (câteva epoci)
Dezghețăm ultimele straturi și facem fine-tuning cu un learning rate foarte mic

Pentru că datasetul e dezechilibrat (sunt mai multe cazuri de pneumonie decât normale), am folosit class weights ca modelul să nu ignore clasa minoritară.

Structura repo-ului
matlab/
    clasificare_pneumonie.m

python/
    resnet18_colab.ipynb
    resnet50_colab.ipynb

results/
    matrice_confuzie_resnet18.png
    matrice_confuzie_resnet50.png
    grafic_performanta.png

Cum rulezi
MATLAB: descarcă datasetul de pe Kaggle, schimbă datasetPath din script cu calea ta și rulează în MATLAB R2021a sau mai nou.
Python (Colab): încarcă ZIP-ul datasetului în Google Drive, deschide notebook-ul în Colab și rulează celulele pe rând. Modelul se salvează automat în Drive la final.
