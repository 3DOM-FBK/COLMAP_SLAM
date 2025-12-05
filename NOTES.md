# Improving BA

`src/odometry/custom_bundle_adjustment.py`
```
 def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
+    #print(ba_options); print(ba_config)
+    #ba_options.use_gpu = True
+    #ba_options.min_num_images_gpu_solver = 5
```

`src/odometry/odometry.py` and `src/odometry/custom_incremental_pipeline.py`
Added `self.run_BA` to run BA only when the last image of the rig is added


Da fare:
* con lo script dei test generare i dataset e modificare cslam per essere piu' robusto
* saefty cage deve passere un parametro in `controls` per dichiarare costantemente qual e' da considerarsi la camera master (questo permette anche la ri-inizializzazione con una certa camera master)
* esportare la covarianza, altri parametri interessanti?
* Modificare il bundle con un custum BA che aggiunga il constraint sulla posa relativa
* passaggio da stereo a monoculsr
* SAREBBE DA FARE LA RESECTION PER TUTTI I FRAMES... how to do?


Implementato:
* Puo' essere passato il comando di re-inizializzazione in qualsiasi momento
* Reso disponibili molti parametri bundle nel file di configurazione esterno
* Aggiunto early stoppin del BA guardando la variazione di costo
* Aggiunto il LOG con tante informazioni interessanti per il safety cage, come il numero di local features estratte






*** cmq se si rompe una camera possono sempre re-inizializzare mettendo come camera master l'altra e andare con un monocular (non so se il fuser lo supporta)
*** potrei passare in output la varianza della baseline calcolata, cosi' che si puo' capire che c'e' un probelma grazie al safety cage ***
*** il fuser puo' funzionare senza scala? oppure possiamo prendere la scala dalla storia passata ***
*** il safety cage durante la re-inizializzazione puo' decidere chi e' la camera master, soprattutto se ci sono stati problemi, e passare a monocular ***
FONDAMENTALE, passare la baseline stimata, monocular/stereo, master camera, covarinace?
* andare a 10 Hz ???
* potremmo comunque calcolare la baseline e poi applicare constraint sulla posa relativa



import gc
gc.collect()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'