/*
 *  training.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 * l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 * gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funcions per entrenar la xarxa neuronal.
 *
 * TODO: Comentar bé aquestes funcions perquè són les que han de
 * paral·lelitzar!!
 */

#include "training.h"

#include <math.h>
#include <stdio.h>


/* Loss accumulated on the GPU during one epoch (see loss_reset/loss_get). */
static float loss_acc[1];
#pragma acc declare create(loss_acc)

/**
 * @brief Inicialitza la capa incial de la xarxa (input layer) amb l'entrada
 *que volem reconeixer.
 *
 * @param i Índex de l'element del conjunt d'entrenament que farem servir.
 **/
void feed_input(int i) {
  // lay, input and num_neurons are not on the GPU (only their arrays are),
  // so we use local pointers to the arrays, like in update_weights.
  const int n = num_neurons[0];
  float *a = lay[0].actv;
  const char *in = input[i];

#pragma acc parallel loop present(a[0 : n], in[0 : n]) async(1)
  for (int j = 0; j < n; j++)
    a[j] = in[j];

  // No wait here: the host only queues the kernels on queue 1 and goes on.
  // The GPU runs them in order. The host waits only in loss_get (once per
  // epoch) and in test_nn.
}

/**
 * @brief out_weights -= alpha * dw and bias -= alpha * dbias, in place on the
 *        GPU. Every element is independent.
 */
void update_weights(void) {
  for (int i = 0; i < num_layers - 1; i++) {
    const int n = num_neurons[i];
    const int m = num_neurons[i + 1] * n;
    float *w = lay[i].out_weights;
    const float *dw = lay[i].dw;
    float *b = lay[i].bias;
    const float *dbias = lay[i].dbias;

    // dw and dbias are now computed on the GPU by back_prop, so they are
    // already there: no update device is needed.
#pragma acc parallel present(w[0 : m], dw[0 : m], b[0 : n], dbias[0 : n])      \
    async(1)
    {
#pragma acc loop gang vector
      for (int j = 0; j < m; j++)
        w[j] = w[j] - (alpha * dw[j]);
#pragma acc loop gang vector
      for (int j = 0; j < n; j++)
        b[j] = b[j] - (alpha * dbias[j]);
    }
  }
  // No wait here: the host only queues the kernels on queue 1 and goes on.
  // The GPU runs them in order. The host waits only in loss_get (once per
  // epoch) and in test_nn.
}

/**
 * @brief Inicialitza les matrius de variacions de pesos (dw) i els vectors de
 * variació de bias (dbias) de totes les capes.
 *
 * @param Això cal fer-ho per cada batch, així es poden acumular les
 *variacions per tots els patrons del batch i aplicar-les a l'acabar de
 *processar tot el batch.
 **/
void init_deltas() {
  for (int i = 0; i < num_layers - 1; i++) {
    // Local pointers to the arrays of layer i (same idea as update_weights).
    const int n = num_neurons[i];
    const int m = num_neurons[i + 1] * n;
    float *dbias = lay[i].dbias;
    float *dw = lay[i].dw;

    // Every element is set to 0 independently, so both loops are parallel.
#pragma acc parallel present(dbias[0 : n], dw[0 : m]) async(1)
    {
#pragma acc loop gang vector
      for (int j = 0; j < n; j++)
        dbias[j] = 0.0;
#pragma acc loop gang vector
      for (int j = 0; j < m; j++)
        dw[j] = 0.0;
    }
  }
  // No wait here: the host only queues the kernels on queue 1 and goes on.
  // The GPU runs them in order. The host waits only in loss_get (once per
  // epoch) and in test_nn.
}

/**
 * @brief Propagació dels valors de les neurones de l'entrada (valors a la
 * input layer) a la resta de capes de la xarxa fins a obtenir una predicció
 * (sortida)
 *
 * @details La capa d'entrada (input layer = capa 0) ja ha estat inicialitzada
 * amb els valors de l'entrada que volem reconeixer. Així, el for més extern
 * (sobre i) recorre totes les capes de la xarxa a partir de la primera capa
 * hidden (capa 1). El for intern (sobre j) recorre les neurones de la capa i
 * calculant el seu valor d'activació [lay[i].actv[j]]. El valor d'activació
 * de cada neurona depén de l'exitació de la neurona calculada en el for més
 * intern (sobre k) [lay[i].z[j]]. El valor d'exitació s'inicialitza amb el
 * biax de la neurona corresponent [j] (lay[i].bias[j]) i es calcula
 * multiplicant el valor d'activació de les neurones de la capa anterior (i-1)
 * pels pesos de les connexions (out_weights) entre les dues capes. Finalment,
 * el valor d'activació de la neurona (j) es calcula fent servir la funció
 * RELU (REctified Linear Unit) si la capa (j) és una capa oculta (hidden) o
 * la funció Sigmoid si es tracte de la capa de sortida.
 *
 */
void forward_prop() {
  // The layers (i) stay sequential: layer i needs the activations of layer i-1.
  for (int i = 1; i < num_layers; i++) {
    // Local pointers to the arrays used by layer i (same idea as
    // update_weights). w holds the weights from layer i-1 to layer i.
    const int n = num_neurons[i];
    const int nprev = num_neurons[i - 1];
    const int is_output = (i == num_layers - 1);
    float *z = lay[i].z;
    float *a = lay[i].actv;
    const float *b = lay[i].bias;
    const float *w = lay[i - 1].out_weights;
    const float *aprev = lay[i - 1].actv;

    // Each neuron j has its own z[j] and a[j], so the neurons are independent.
    // v2: each neuron j gets a whole block of threads ("gang"), and the 128
    // threads of the block ("vector") split the sum over k and add their
    // partial sums at the end ("reduction"). This gives many more threads than
    // one thread per neuron, and neighbour threads read neighbour positions of
    // w (row j), which is the fastest way to read GPU memory.
    // (The debug prints of the original code were removed: printf cannot run
    // on the GPU.)
#pragma acc parallel loop gang                                                 \
    present(z[0 : n], a[0 : n], b[0 : n], w[0 : n * nprev], aprev[0 : nprev])  \
    async(1)
    for (int j = 0; j < n; j++) {
      float sum = b[j];
#pragma acc loop vector reduction(+ : sum)
      for (int k = 0; k < nprev; k++)
        sum += w[j * nprev + k] * aprev[k];
      z[j] = sum;
      if (!is_output) // Relu Activation Function for Hidden Layers
        a[j] = (sum < 0) ? 0 : sum;
      else // Sigmoid Activation Function for Output Layer
        a[j] = 1 / (1 + exp(-sum));
    }
  }
  // No wait here: the host only queues the kernels on queue 1 and goes on.
  // The GPU runs them in order. The host waits only in loss_get (once per
  // epoch) and in test_nn.
}

/**
 * @brief Calcula la funció de perdua per cada patró i l'acumulat fins el punt
 * actual d'entrenament
 *
 * @details La funció de pèrdua (loss function) en l'entrenament
 * d'una xarxa neuronal multivivell és essencialment una mesura de com de
 * malament li va a la xarxa per realitzar la tasca per a la qual està sent
 * entrenada, donades les seves sortides actuals en comparació amb les
 * sortides desitjades (ground truth). Fem servir la funció de loss Entropia
 * Creuada Binària (Binary Cross-Entropy o Log Loss) que s'utilitza per a
 * problemes de classificació quan la capa de sortida fa servir una funció
 * d'activació sigmoide.
 */
float compute_loss(int i) {
  // Local pointers: output activations and desired outputs of pattern i.
  const int n = num_neurons[num_layers - 1];
  const float *a = lay[num_layers - 1].actv;
  const float *d = desired_outputs[i];

  // v2: the profile showed that one thread doing the 10 logs one after the
  // other was slow (about 15 us per pattern). Now there is ONE block
  // (num_gangs(1)) and its threads ("vector") do one output neuron each; their
  // partial sums are added with a reduction. The loss of the pattern is then
  // added to loss_acc, which lives on the GPU, so nothing goes to the host.
  // num_gangs(1) is needed: with more blocks, every block would add its own
  // copy of the loss to loss_acc.
#pragma acc parallel num_gangs(1) vector_length(32)                           \
    present(a[0 : n], d[0 : n], loss_acc) async(1)
  {
    double tmpcost = 0;
#pragma acc loop vector reduction(+ : tmpcost)
    for (int j = 0; j < n; j++) {
      double p = a[j];
      if (a[j] < EPSILON)
        p = EPSILON;
      else if (a[j] > 1 - EPSILON)
        p = 1 - EPSILON;

      tmpcost += d[j] * log(p) + (1 - d[j]) * log(1.0 - p);
    }
    float loss = -tmpcost / n;
    loss_acc[0] += loss;
  }
  // The loss stays on the GPU (read once per epoch with loss_get), so there is
  // no value to return here. The return type is kept to not change training.h.
  return 0.0f;
}

/**
 * @brief Sets the epoch loss accumulator to 0 on the host and on the GPU.
 *        Called at the start of every epoch from train_neural_net.
 */
void loss_reset(void) {
  loss_acc[0] = 0.0f;
#pragma acc update device(loss_acc) async(1)
}

/**
 * @brief Waits for the GPU and brings back the loss accumulated during the
 *        epoch (one float per epoch). Called at the end of every epoch from
 *        train_neural_net.
 */
float loss_get(void) {
#pragma acc update host(loss_acc) async(1)
#pragma acc wait(1)
  return loss_acc[0];
}

/**
 * @brief Calcula el gradient que es necessari aplicar als pesos de les
 * connexions entre neurones per corregir els errors de predicció
 *
 * @details Calcula dos vectors de correcció per cada capa de la xarxa, un per
 * corregir els pesos de les connexions de la neurona (j) amb la capa anterior
 *          (lay[i-1].dw[j]) i un segon per corregir el biax de cada neurona
 * de la capa actual (lay[i].bias[j]). Hi ha un tractament diferent per la
 * capa de sortida (num_layesr -1) perquè aquest és l'única cas en el que
 * l'error es conegut (lay[num_layers-1].actv[j] - desired_outputs[p][j]).
 * Això es pot veure en els dos primers fors. Per totes les capes ocultes
 * (hidden layers) no es pot saber el valor d'activació esperat per a cada
 * neurona i per tant es fa una estimació. Aquest càlcul es fa en el doble for
 * que recorre totes les capes ocultes (sobre i) neurona a neurona (sobre j).
 * Es pot veure com en cada cas es fa una estimació de quines hauríen de ser
 * les activacions de les neurones de la capa anterior (lay[i-1].dactv[k] =
 * lay[i-1].out_weights[j* num_neurons[i-1] + k] * lay[i].dz[j];), excepte pel
 * cas de la capa d'entrada (input layer) que és coneguda (imatge d'entrada).
 *
 */
void back_prop(int p) {
  // The original double loop (j outer, k inner) can not be parallelized over j
  // as it is: every j adds into the same dactv[k] (race condition). So each
  // layer is split into three simple loops, all on the GPU queue 1:
  //   A) loop over j: dz[j] and dbias[j]          (one writer per j)
  //   B) loops over j and k: dw[j * nprev + k]    (one writer per element)
  //   C) loop over k: dactv[k] = sum over j        (one writer per k)
  // In C each thread k does its own sum over j, in the same order as the
  // original code, so there is no race and no atomic is needed.

  // ---- Output layer (L) and the weights from layer L-1 to L ----
  const int L = num_layers - 1;
  const int nout = num_neurons[L];
  const int nprev = num_neurons[L - 1];
  const float *a_out = lay[L].actv;
  float *dz_out = lay[L].dz;
  float *dbias_out = lay[L].dbias;
  const float *d = desired_outputs[p];
  float *dw = lay[L - 1].dw;
  const float *w = lay[L - 1].out_weights;
  const float *aprev = lay[L - 1].actv;
  float *dactv_prev = lay[L - 1].dactv;

  // A) Error of every output neuron (the only layer where it is known).
#pragma acc parallel loop                                                      \
    present(a_out[0 : nout], dz_out[0 : nout], dbias_out[0 : nout],            \
                d[0 : nout]) async(1)
  for (int j = 0; j < nout; j++) {
    dz_out[j] = (a_out[j] - d[j]) * a_out[j] * (1 - a_out[j]);
    dbias_out[j] += dz_out[j];
  }

  // B) Weight gradients: every (j, k) element is independent.
#pragma acc parallel loop collapse(2)                                          \
    present(dw[0 : nout * nprev], dz_out[0 : nout], aprev[0 : nprev]) async(1)
  for (int j = 0; j < nout; j++)
    for (int k = 0; k < nprev; k++)
      dw[j * nprev + k] += dz_out[j] * aprev[k];

  // C) Estimated error of the previous layer: one thread per k.
  // "gang vector" on k: one GPU thread per k. "loop seq" on j: each thread
  // does its whole sum over j by itself, in order (without these clauses the
  // compiler splits j among threads with a reduction, which changes the order
  // of the sums).
#pragma acc parallel loop gang vector                                          \
    present(dactv_prev[0 : nprev], w[0 : nout * nprev], dz_out[0 : nout])      \
    async(1)
  for (int k = 0; k < nprev; k++) {
    float sum = 0;
#pragma acc loop seq
    for (int j = 0; j < nout; j++)
      sum += w[j * nprev + k] * dz_out[j];
    dactv_prev[k] = sum;
  }

  // ---- Hidden layers, from the last one back to layer 1 (sequential:
  // layer i-1 needs the dactv computed for layer i) ----
  for (int i = num_layers - 2; i > 0; i--) {
    const int n = num_neurons[i];
    const int np = num_neurons[i - 1];
    const float *z = lay[i].z;
    const float *dactv = lay[i].dactv;
    float *dz = lay[i].dz;
    float *dbias = lay[i].dbias;
    float *dwh = lay[i - 1].dw;
    const float *wh = lay[i - 1].out_weights;
    const float *ap = lay[i - 1].actv;
    float *dap = lay[i - 1].dactv;

    // A) dz and dbias of every neuron j of layer i (ReLU derivative).
#pragma acc parallel loop                                                      \
    present(z[0 : n], dactv[0 : n], dz[0 : n], dbias[0 : n]) async(1)
    for (int j = 0; j < n; j++) {
      dz[j] = (z[j] >= 0) ? dactv[j] : 0;
      dbias[j] += dz[j];
    }

    // B) Weight gradients from layer i-1 to layer i.
#pragma acc parallel loop collapse(2)                                          \
    present(dwh[0 : n * np], dz[0 : n], ap[0 : np]) async(1)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < np; k++)
        dwh[j * np + k] += dz[j] * ap[k];

    // C) Estimated error of layer i-1 (not needed for the input layer).
    if (i > 1) {
      // Same clauses as loop C of the output layer: one thread per k, the sum
      // over j done in order inside each thread.
#pragma acc parallel loop gang vector                                          \
    present(dap[0 : np], wh[0 : n * np], dz[0 : n]) async(1)
      for (int k = 0; k < np; k++) {
        float sum = 0;
#pragma acc loop seq
        for (int j = 0; j < n; j++)
          sum += wh[j * np + k] * dz[j];
        dap[k] = sum;
      }
    }
  }
}
