#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "genbrush.h"
#include "gradautomaton.h"
#include "genalg.h"

// Nb of step between each save of the GenAlg
// Saving it allows to restart a stop learning process but is
// very time consuming if there are many input/hidden/output
// If 0 never save
#define SAVE_GA_EVERY 0

// Size of the gene pool and elite pool
#define ADN_SIZE_POOL 20
#define ADN_SIZE_ELITE 10

// Diversity threshold for KT event in GenAlg
#define DIVERSITY_THRESHOLD 0.00001

// Initial best value during learning, must be lower than any
// possible value returned by GANNIEvaluate()
#define INIT_BEST_VAL -10000.0

// Value of the NeuraNet above which the learning process stops
// In percentage of similarity
#define STOP_LEARNING_AT_VAL 0.99

// Number of epoch above which the learning process stops
#define STOP_LEARNING_AT_EPOCH 100000

// Save NeuraNet in compact format
#define COMPACT true

// Flag to display the TextOMeter
#define FLAG_TEXTOMETER true

// Number of dimensions in addition to the RGBA
#define STATUS_EXTRA_DIM 2

// ------------- GradAutomatonNeuraNetImg

// ================= Data structure ===================

typedef enum GANNImode {

  GANNInothing,
  GANNIencoding,
  GANNIdecoding

} GANNImode;

typedef struct GANNI {

  // Mode of the GANNI
  GANNImode mode;

  // Flag to mnemorize the verbose mode
  bool verbose;

  // Path to the image to encode or ganni file to decode
  char* path;

  // Dimension of the status of the cells
  // Must be > 4 (RGBA + data)
  long dimStatus;

  // Number of hidden layers in the NeuraNet
  long nbHiddenLayers;

  // GradAutomatonNeuraNet used to encode/decode
  GradAutomatonNeuraNet* gann;

  // Target/result GenBrush
  GenBrush* gb;

  // Path to the result of encoding
  char* outputPath;

int i;

} GANNI;

// ================ Functions declaration ====================

// Create a new GANNI
GANNI GANNICreateStatic(void);

// Free the memory used by the static GANNI 'that'
void GANNIFreeStatic(GANNI* const that);

// Get the mode of the GANNI 'that'
static inline
GANNImode GANNIGetMode(const GANNI* const that);

// Set the mode of the GANNI 'that' to 'mode'
static inline
void GANNISetMode(
     GANNI* const that,
  const GANNImode mode);

// Get the path of the GANNI 'that'
static inline
const char* GANNIPath(const GANNI* const that);

// Set the path of the GANNI 'that' to 'path'
static inline
void GANNISetPath(
  GANNI* const that,
         char* path);

// Get the verbose of the GANNI 'that'
static inline
bool GANNIGetVerbose(const GANNI* const that);

// Set the verbose of the GANNI 'that' to 'verbose'
static inline
void GANNISetVerbose(
  GANNI* const that,
    const bool verbose);

// Get the dimStatus of the GANNI 'that'
static inline
long GANNIGetDimStatus(const GANNI* const that);

// Get the nbHiddenLayers of the GANNI 'that'
static inline
long GANNIGetNbHiddenLayers(const GANNI* const that);

// Set the dimStatus of the GANNI 'that' to 'dimStatus'
static inline
void GANNISetDimStatus(
  GANNI* const that,
    const long dimStatus);

// Set the nbHiddenLayers of the GANNI 'that' to 'nbHiddenLayers'
static inline
void GANNISetNbHiddenLayers(
  GANNI* const that,
    const long nbHiddenLayers);

// Process arguments from the command line
// Return true if the processing has been succesful, false else
bool GANNIProcessArg(
  GANNI* const that,
     const int argc,
  char** const argv);

// Run the encoding process for the GANNI 'that'
void GANNIEncode(GANNI* const that);

// Train the NeuraNet of the GradAutomatonNeuraNet 'that->grad'
// of the GANNI 'that' on the GenBrush 'that->gb'
void GANNITrain(GANNI* const that);

// Run the decoding process for the GANNI 'that'
void GANNIDecode(GANNI* const that);

// Return the similarity of the GenBrush produced by the
// GradAutomatonNeuraNet of 'that' with the GenBrush of 'that'
float GANNIEvaluate(
                  GANNI* const that,
                   const float curWorstVal);

// Return the image produced by the GradAutomatonNeuraNet of 'that'
// as a GenBrush
GenBrush* GANNI2GB(GANNI* const that);

// ================ Functions implementation ====================

// Create a new GANNI
GANNI GANNICreateStatic(void) {

  // Declare the new GANNI
  GANNI that;

  // Init properties
  that.mode = GANNInothing;
  that.path = NULL;
  that.verbose = false;
  that.dimStatus = 4 + STATUS_EXTRA_DIM;
  that.nbHiddenLayers = 1;
  that.gann = NULL;
  that.gb = NULL;
  that.outputPath = NULL;

  // Return the new GANNI
  return that;

}

// Free the memory used by the static GANNI 'that'
void GANNIFreeStatic(GANNI* const that) {

  // Nothing to do
  (void)that;
  return;

}

// Get the mode of the GANNI 'that'
static inline
GANNImode GANNIGetMode(const GANNI* const that) {

  // Return the mode
  return that->mode;

}

// Set the mode of the GANNI 'that' to 'mode'
static inline
void GANNISetMode(
     GANNI* const that,
  const GANNImode mode) {

  // Set the mode
  that->mode = mode;

}

// Get the path of the GANNI 'that'
static inline
const char* GANNIPath(const GANNI* const that) {

  // Return the path
  return that->path;

}

// Set the path of the GANNI 'that' to 'path'
static inline
void GANNISetPath(
  GANNI* const that,
         char* path) {

  // Set the path
  that->path = path;

}

// Get the verbose of the GANNI 'that'
static inline
bool GANNIGetVerbose(const GANNI* const that) {

  // Return the verbose flag
  return that->verbose;

}

// Set the verbose of the GANNI 'that' to 'verbose'
static inline
void GANNISetVerbose(
  GANNI* const that,
    const bool verbose) {

  // Set the verbose flag
  that->verbose = verbose;

}

// Get the dimStatus of the GANNI 'that'
static inline
long GANNIGetDimStatus(const GANNI* const that) {

  // Return the dimStatus
  return that->dimStatus;

}

// Get the nbHiddenLayers of the GANNI 'that'
static inline
long GANNIGetNbHiddenLayers(const GANNI* const that) {

  // Return the dimStatus
  return that->nbHiddenLayers;

}

// Set the dimStatus of the GANNI 'that' to 'dimStatus'
static inline
void GANNISetDimStatus(
  GANNI* const that,
    const long dimStatus) {

  // Set the dimStatus
  that->dimStatus = dimStatus;

}

// Set the nbHiddenLayers of the GANNI 'that' to 'dimStatus'
static inline
void GANNISetNbHiddenLayers(
  GANNI* const that,
    const long nbHiddenLayers) {

  // Set the nbHiddenLayers
  that->nbHiddenLayers = nbHiddenLayers;

}

// Process arguments from the command line
// Return true if the processing has been succesful, false else
bool GANNIProcessArg(
  GANNI* const that,
     const int argc,
  char** const argv) {

  // Declare a flag to memorize the success of processing
  bool flag = true;

  // Declare a variable to process the arguments
  int match = true;

  // Loop on arguments
  for (
    int iArg = 0;
    (iArg < argc) && (flag == true);
    ++iArg) {

    // Help
    match =
      strcmp(
        argv[iArg],
        "-help");
    if (match == 0) {

      printf(
        "ganni [-help] " \
        "[-encode <img.tga>] " \
        "[-decode <file.gni>] " \
        "[-verbose] " \
        "[-dimStatus <dim, int > 4, by def. 5>] " \
        "[-nbHiddenLayers <nb, int >= 0, by def. 1>] " \
        "\n");

    }

    // Encoding
    match =
      strcmp(
        argv[iArg],
        "-encode");
    if (match == 0) {

      // If the user has provided a path
      if (iArg < argc - 1) {

        // Set the mode
        GANNISetMode(
          that,
          GANNIencoding);

        // Move to the next argument
        ++iArg;

        // Memorize the path to the image
        GANNISetPath(
          that,
          argv[iArg]);

      // Else the user hasn't provided a path
      } else {

        // Update the flag
        flag = false;

      }

    }

    // Decoding
    match =
      strcmp(
        argv[iArg],
        "-decode");
    if (match == 0) {

      // If the user has provided a path
      if (iArg < argc - 1) {

        // Set the mode
        GANNISetMode(
          that,
          GANNIdecoding);

        // Move to the next argument
        ++iArg;

        // Memorize the path to the image
        GANNISetPath(
          that,
          argv[iArg]);

      // Else the user hasn't provided a path
      } else {

        // Update the flag
        flag = false;

      }

    }

    // Verbose
    match =
      strcmp(
        argv[iArg],
        "-verbose");
    if (match == 0) {

      GANNISetVerbose(
        that,
        true);

    }

    // Set the dimStatus
    match =
      strcmp(
        argv[iArg],
        "-dimStatus");
    if (match == 0) {

      // If the user has provided a dimStatus
      if (iArg < argc - 1) {

        // Move to the next argument
        ++iArg;

        // Set the dimStatus
        GANNISetDimStatus(
          that,
          atol(argv[iArg]));

      // Else the user hasn't provided a dimStatus
      } else {

        // Update the flag
        flag = false;

      }

    }

    // Set the nbHiddenLayers
    match =
      strcmp(
        argv[iArg],
        "-nbHiddenLayers");
    if (match == 0) {

      // If the user has provided a nbHiddenLayers
      if (iArg < argc - 1) {

        // Move to the next argument
        ++iArg;

        // Set the nbHiddenLayers
        GANNISetNbHiddenLayers(
          that,
          atol(argv[iArg]));

      // Else the user hasn't provided a nbHiddenLayers
      } else {

        // Update the flag
        flag = false;

      }

    }

  }

  // Return the flag
  return flag;

}

// Run the encoding process for the GANNI 'that'
void GANNIEncode(GANNI* const that) {

  // Open the image
  that->gb = GBCreateFromFile(GANNIPath(that));

  if (that->gb == NULL) {

    printf(
      "Could not open the image [%s]\n",
      GANNIPath(that));
    return;

  } else {

    if (GANNIGetVerbose(that) == true) {

      printf(
        "Encoding [%s]\n",
        GANNIPath(that));

    }

  }

  // Create the path for the output GANNI file
  that->outputPath = strdup(GANNIPath(that));
  size_t length = strlen(that->outputPath);
  that->outputPath[length - 3] = 'g';
  that->outputPath[length - 2] = 'n';
  that->outputPath[length - 1] = 'i';
  if (GANNIGetVerbose(that) == true) {

    printf(
      "Saving to [%s]\n",
      that->outputPath);

  }

  // Get the dimension of the image
  VecShort2D* dimImg = GBDim(that->gb);
  if (GANNIGetVerbose(that) == true) {

    printf("Image dimensions: ");
    VecPrint(
      dimImg,
      stdout);
    printf("\n");

  }

  // Create the GradAutomatonNeuraNet
  bool diagLink = true;
  that->gann =
    GradAutomatonCreateNeuraNetSquare(
      GANNIGetDimStatus(that),
      dimImg,
      diagLink,
      GANNIGetNbHiddenLayers(that));

  // Train the NeuraNet
  GANNITrain(that);

  // Free memory
  free(that->outputPath);
  GradAutomatonNeuraNetFree(&(that->gann));
  GBFree(&(that->gb));

  // Display a message for the user
  if (GANNIGetVerbose(that) == true) {

    printf("Encoding ended\n");

  }

}

// Return the image produced by the GradAutomatonNeuraNet of 'that'
// as a GenBrush
GenBrush* GANNI2GB(GANNI* const that) {

  // Declare a variable to memorize the dimension of the Grad
  const VecShort2D* dim =
    GradDim(GradAutomatonGrad(that->gann));

  // Declare a variable to memorize the position at the center of the
  // Grad
  VecShort2D posCenter =
    VecShortCreateStatic2D();
  for (
    int iDim = 2;
    iDim--;) {

    short v =
      VecGet(
        dim,
        iDim);

    VecSet(
      &posCenter,
      iDim,
      v / 2);

    }

  // Loop on the cell of the Grad
  VecShort2D pos =
    VecShortCreateStatic2D();
  bool flagStep = true;
  do {

    // Get the current and previous status of the cell
    // at the current position
    GrACellFloat* cell =
      GradAutomatonCell(
        that->gann,
        &pos);
    VecFloat* prevStatus =
      GrACellPrevStatus(cell);
    VecFloat* curStatus =
      GrACellCurStatus(cell);

    // Declare a variable to memorize the initial rgba values of the
    // status, by default transparent black
    float val = 0.0;

    // If the cell is at the center
    bool isAtCenter =
      VecIsEqual(
        &pos,
        &posCenter);
    if (isAtCenter == true) {

      // By default the rgba value of the center pixel is opaque white
      val = 1.0;

    }

    // Set the values of the previous status of the cell
    // The values of the extra dimensions are set by default to the
    // same values as the rgba of the cell
    for (
      int iDim = VecGetDim(curStatus);
      iDim--;) {

      VecSet(
        prevStatus,
        iDim,
        val);
      VecSet(
        curStatus,
        iDim,
        val);

    }

    // Step to the next cell
    flagStep =
      VecStep(
        &pos,
        dim);

  } while (flagStep);

  // Calculate the maximum nb of steps to reach stability
  // It is the maximum dimension of the Grad of gann
  long nbMaxStep = VecGetMaxVal(dim);

  // Step the GradAutomaton until stability or maximum nb of steps is
  // reached
  long iStep = 0;
  do {

    // Step the GradAutomaton
    GradAutomatonStep(that->gann);

    // Increment the nb of steps
    ++iStep;

  } while (
    GradAutomatonIsStable(that->gann) == false &&
    iStep < nbMaxStep);

  // Create the result GenBrush
  GenBrush* gb = GBCreateImage(dim);

  // Loop on the cell of the Grad to convert it into a GenBrush
  VecSetNull(&pos);
  do {

    // Get the current status of the cell at the current position
    GrACellFloat* cell =
      GradAutomatonCell(
        that->gann,
        &pos);
    VecFloat* status = GrACellCurStatus(cell);

    // Declare a GBPixel to do the conversion
    GBPixel pixel;

    // Convert the values of the current status of the cell
    // into pixel values
    for (
      int iDim = 4;
      iDim--;) {

      float val =
        VecGet(
          status,
          iDim);
      val =
        MIN(
          255.99,
          MAX(
            0.0,
            val));
      pixel._rgba[iDim] = (unsigned char)val;

    }

    // Update the pixel in the GenBrush
    GBSetFinalPixel(
      gb,
      &pos,
      &pixel);

    // Step to the next cell
    flagStep =
      VecStep(
        &pos,
        dim);

  } while (flagStep);

  // Return the GenBrush
  return gb;

}

// Return the similarity of the GenBrush produced by the
// GradAutomatonNeuraNet of 'that' with the GenBrush of 'that'
float GANNIEvaluate(
  GANNI* const that,
   const float curWorstVal) {

  // Can't use the current worst val to optimize in the case of GANNI
  (void)curWorstVal;

  // Create the GenBrush from the GradAutomaton
  GenBrush* gbGrA = GANNI2GB(that);

  // Compare the GenBrush to the learnt one
  /*float similarity =
    GBGetSimilarity(
      that->gb,
      gbGrA);*/

  // Declare a variable to memorize the dimension of the Grad
  const VecShort2D* dim =
    GradDim(GradAutomatonGrad(that->gann));

  // Declare a variable to memorize the number of identic pixels
  float nbSamePix = 0.0;

  // Loop on the pixels
  VecShort2D pos =
    VecShortCreateStatic2D();
  bool flagStep = true;
  do {

    // Declare a GBPixel to do the conversion
    GBPixel pixelGB =
      GBGetFinalPixel(
        that->gb,
        &pos);
    GBPixel pixelGBGrA =
      GBGetFinalPixel(
        gbGrA,
        &pos);

    // Compare the pixels
    bool flagSame = true;
    for (
      int iDim = 4;
      iDim-- && flagSame == true;) {

      if (pixelGB._rgba[iDim] != pixelGBGrA._rgba[iDim]) {

        flagSame = false;

      }

    }

    // Update the number of identic pixels
    if (flagSame == true) {

      nbSamePix += 1.0;

    }

    // Step to the next cell
    flagStep =
      VecStep(
        &pos,
        dim);

  } while (flagStep);

  // Calculate the similarity
  float similarity = nbSamePix / (float)GBArea(that->gb);

//if (similarity > curWorstVal) {
char fn[17];
sprintf(fn,"./ganni2gb%02d.tga",that->i);
GBSetFileName(gbGrA, fn);
GBRender(gbGrA);
//}

  // Return the result of evaluation
  return similarity;

}

// Train the NeuraNet of the GradAutomatonNeuraNet 'that->grad'
// of the GANNI 'that' on the GenBrush 'that->gb'
void GANNITrain(GANNI* const that) {

  // Init the random generator
  srandom(time(NULL));

  // Declare variables to measure time
  struct timespec start;
  struct timespec stop;

  // Start measuring time
  clock_gettime(
    CLOCK_REALTIME,
    &start);

  // Get a reference to the NeuraNet of the gann
  NeuraNet* nn =
    GrAFunNeuraNetNN((GrAFunNeuraNet*)GradAutomatonFun(that->gann));

  // Declare a variable to memorize the best value
  float bestVal = INIT_BEST_VAL;

  // Declare a variable to memorize the limit in term of epoch
  unsigned long int limitEpoch = STOP_LEARNING_AT_EPOCH;

  // Greate the GenAlg
  GenAlg* ga =
    GenAlgCreate(
      ADN_SIZE_POOL,
      ADN_SIZE_ELITE,
      NNGetGAAdnFloatLength(nn),
      NNGetGAAdnIntLength(nn));
  NNSetGABoundsBases(
    nn,
    ga);
  NNSetGABoundsLinks(
    nn,
    ga);
  GASetTypeNeuraNet(
    ga,
    NNGetNbInput(nn),
    NNGetNbMaxHidden(nn),
    NNGetNbOutput(nn));
  GASetNeuraNetLinkMutability(
    ga,
    false);
  GASetDiversityThreshold(
    ga,
    DIVERSITY_THRESHOLD);
  GAInit(ga);
  
  GASetTextOMeterFlag(
    ga,
    FLAG_TEXTOMETER);

  // Start learning process
  if (GANNIGetVerbose(that) == true) {

    printf("Learning...\n");
    printf("Will stop when curEpoch >= %lu or bestVal >= %f\n",
      limitEpoch,
      STOP_LEARNING_AT_VAL);
    fflush(stdout);

  }

  // Declare a variable to memorize the best value in the current epoch
  float curBest = 0.0;
  float curWorst = 0.0;
  float curWorstElite = 0.0;

  // Declare a variable to manage the save of GenAlg
  int delaySave = 0;

  // Learning loop
  while (
    bestVal < STOP_LEARNING_AT_VAL &&
    GAGetCurEpoch(ga) < limitEpoch) {

    curWorst = curBest;
    curBest = INIT_BEST_VAL;
    curWorstElite = INIT_BEST_VAL;
    int curBestI = 0;

    // For each adn in the GenAlg
    int startEnt = 0;
    if (
      GAGetCurEpoch(ga) > 0 &&
      GAGetFlagKTEvent(ga) == false) {

      startEnt = GAGetNbElites(ga);

    }

    for (
      int iEnt = startEnt;
      iEnt < GAGetNbAdns(ga);
      ++iEnt) {

      // Get the adn
      GenAlgAdn* adn =
        GAAdn(
          ga,
          iEnt);

      // Set the links and base functions of the NeuraNet according
      // to this adn
      if (GABestAdnF(ga) != NULL) {

        NNSetBases(
          nn,
          GAAdnAdnF(adn));

      }

      // Evaluate the NeuraNet
that->i=iEnt;
      float value =
        GANNIEvaluate(
          that,
          curWorstElite);

      // Update the value of this adn
      GASetAdnValue(
        ga,
        adn,
        value);

      // Update the best value in the current epoch
      if (value > curBest) {

        curBest = value;
        curBestI = iEnt;

      }

      if (value < curWorst) {

        curWorst = value;

      }

    }

    // Memorize the current value of the worst elite
    GenAlgAdn* worstEliteAdn =
      GAAdn(
        ga,
        GAGetNbElites(ga) - 1);
    curWorstElite = GAAdnGetVal(worstEliteAdn);

    // Measure time
    clock_gettime(
      CLOCK_REALTIME,
      &stop);
    float elapsed = stop.tv_sec - start.tv_sec;
    int day = (int)floor(elapsed / 86400);
    elapsed -= (float)(day * 86400);
    int hour = (int)floor(elapsed / 3600);
    elapsed -= (float)(hour * 3600);
    int min = (int)floor(elapsed / 60);
    elapsed -= (float)(min * 60);
    int sec = (int)floor(elapsed);

    // If there has been improvement during this epoch
    if (curBest > bestVal) {

      bestVal = curBest;

      // Display info about the improvment
      if (GANNIGetVerbose(that) == true) {

        printf(
          "Improvement at epoch %05lu: %f(%03d) (in " \
          "%02d:%02d:%02d:%02ds)       \n",
          GAGetCurEpoch(ga),
          bestVal,
          curBestI,
          day,
          hour,
          min,
          sec);
        fflush(stdout);

      }

      // Set the links and base functions of the NeuraNet according
      // to the best adn
      GenAlgAdn* bestAdn =
        GAAdn(
          ga,
          curBestI);
      if (GAAdnAdnF(bestAdn) != NULL) {

        NNSetBases(
          nn,
          GAAdnAdnF(bestAdn));

      }

      // Save the best GradAutomatonNeuraNet
      FILE* fd =
        fopen(
          that->outputPath,
          "w");
      bool retSave =
        GradAutomatonSave(
          that->gann,
          fd,
          COMPACT);
      if (retSave == false) {

        fprintf(
          stderr,
          "Couldn't save the GradAutomatonNeuraNet\n");
        exit(1);

      }

      fclose(fd);

    } else {

      if (GANNIGetVerbose(that) == true) {

        GenAlgAdn* firstAdn =
          GAAdn(
            ga,
            0);

        printf(
          "Epoch %05lu: v%f a%03lu kt%03lu ",
          GAGetCurEpoch(ga),
          GAAdnGetVal(firstAdn),
          GAAdnGetAge(firstAdn),
          GAGetNbKTEvent(ga));
        printf(
          "(in %02d:%02d:%02d:%02ds)  \r",
          day,
          hour,
          min,
          sec);
        fflush(stdout);

      }

    }

    ++delaySave;
    if (
      SAVE_GA_EVERY != 0 &&
      delaySave >= SAVE_GA_EVERY) {

      delaySave = 0;

      // Save the adns of the GenAlg, use a temporary file to avoid
      // loosing the previous one if something goes wrong during
      // writing, then replace the previous file with the temporary one
      FILE* fd =
        fopen(
          "./bestga.tmp",
          "w");

      bool retSave =
        GASave(
          ga,
          fd,
          COMPACT);
      if (retSave == false) {

        fprintf(
          stderr,
          "Couldn't save the GenAlg\n");
        exit(1);

      }

      fclose(fd);
      int ret = system("mv ./bestga.tmp ./bestga.txt");
      (void)ret;

    }

    // Step the GenAlg
    GAStep(ga);

  }

  // Measure time
  clock_gettime(
    CLOCK_REALTIME,
    &stop);
  float elapsed = stop.tv_sec - start.tv_sec;
  int day = (int)floor(elapsed / 86400);
  elapsed -= (float)(day * 86400);
  int hour = (int)floor(elapsed / 3600);
  elapsed -= (float)(hour * 3600);
  int min = (int)floor(elapsed / 60);
  elapsed -= (float)(min * 60);
  int sec = (int)floor(elapsed);
  if (GANNIGetVerbose(that) == true) {

    printf(
      "\nLearning complete (in %d:%d:%d:%ds)\n",
      day,
      hour,
      min,
      sec);
    fflush(stdout);

  }

  // Free memory
  GenAlgFree(&ga);

}

// Run the decoding process for the GANNI 'that'
void GANNIDecode(GANNI* const that) {

  if (GANNIGetVerbose(that) == true) {

    printf(
      "Decoding [%s]\n",
      GANNIPath(that));

  }

  // Load the GradAutomatonNeuraNet
  FILE* fd =
    fopen(
      GANNIPath(that),
      "r");
  bool retLoad =
    GradAutomatonLoad(
      &(that->gann),
      fd);
  if (retLoad == false) {

    fprintf(
      stderr,
      "Couldn't load the GradAutomatonNeuraNet\n");
    exit(1);

  }

  // Create the GenBrush
  GenBrush* gb = GANNI2GB(that);

  // Create the output path
  that->outputPath = strdup(GANNIPath(that));
  size_t length = strlen(that->outputPath);
  that->outputPath[length - 3] = 't';
  that->outputPath[length - 2] = 'g';
  that->outputPath[length - 1] = 'a';
  if (GANNIGetVerbose(that) == true) {

    printf(
      "Saving to [%s]\n",
      that->outputPath);

  }

  // Render the GenBrush ot the file
  GBSetFileName(
    gb,
    that->outputPath);
  GBRender(gb);

  // Free memory
  free(that->outputPath);
  GBFree(&gb);

}

// ------------- Main function

int main(
     int argc,
  char** argv) {

  // Declare the GANNI
  GANNI ganni = GANNICreateStatic();

  // Process the command line arguments
  bool success =
    GANNIProcessArg(
      &ganni,
      argc,
      argv);

  // If the processing was successfull
  if (success == true) {

    // Call the requested processing
    switch(GANNIGetMode(&ganni)) {

      case GANNIencoding:
        GANNIEncode(&ganni);
        break;
      case GANNIdecoding:
        GANNIDecode(&ganni);
        break;
      default:
        break;

    }

  }

  // Free the memory used by the GANNI
  GANNIFreeStatic(&ganni);

  return 0;

}
