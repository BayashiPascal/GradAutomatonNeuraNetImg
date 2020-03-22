#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "genbrush.h"
#include "gradautomaton.h"
#include "genalg.h"

// Size of the gene pool and elite pool
#define ADN_SIZE_POOL 100
#define ADN_SIZE_ELITE 20
// Diversity threshold for KT event in GenAlg
#define DIVERSITY_THRESHOLD 0.00001
// Initial best value during learning, must be lower than any
// possible value returned by Evaluate()
#define INIT_BEST_VAL -10000.0
// Value of the NeuraNet above which the learning process stops
#define STOP_LEARNING_AT_VAL -0.01
// Number of epoch above which the learning process stops
#define STOP_LEARNING_AT_EPOCH 100000

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

// Set the dimStatus of the GANNI 'that' to 'dimStatus'
static inline
void GANNISetDimStatus(
  GANNI* const that,
    const long dimStatus);

// Process arguments from the command line
// Return true if the processing has been succesful, false else
bool GANNIProcessArg(
  GANNI* const that,
     const int argc,
  char** const argv);

// Run the encoding process for the GANNI 'that'
void GANNIEncode(GANNI* const that);

// Train the NeuraNet of the GradAutomatonNeuraNet 'gann'
// for the GANNI 'that' on the GenBrush 'gb'
void GANNITrain(
                  GANNI* const that,
               NeuraNet* const nn,
  GradAutomatonNeuraNet* const gann,
               GenBrush* const gb);

// Run the decoding process for the GANNI 'that'
void GANNIDecode(GANNI* const that);

// ================ Functions implementation ====================

// Create a new GANNI
GANNI GANNICreateStatic(void) {

  // Declare the new GANNI
  GANNI that;

  // Init properties
  that.mode = GANNInothing;
  that.path = NULL;
  that.verbose = false;
  that.dimStatus = 5;

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

// Set the dimStatus of the GANNI 'that' to 'dimStatus'
static inline
void GANNISetDimStatus(
  GANNI* const that,
    const long dimStatus) {

  // Set the dimStatus
  that->dimStatus = dimStatus;

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
        "[-dimStatus]" \
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

  }

  // Return the flag
  return flag;

}

// Run the encoding process for the GANNI 'that'
void GANNIEncode(GANNI* const that) {

  // Open the image
  GenBrush* gb = GBCreateFromFile(GANNIPath(that));

  if (gb == NULL) {

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

  // Get the dimension of the image
  VecShort2D* dimImg = GBDim(gb);
  if (GANNIGetVerbose(that) == true) {

    printf("Image dimensions: ");
    VecPrint(
      dimImg,
      stdout);
    printf("\n");

  }

  // Get the dimension of the input and output of the NeuraNet
  int nbIn = 9 * GANNIGetDimStatus(that);
  int nbOut = GANNIGetDimStatus(that);
  VecLong* hiddenLayers =
    VecLongCreate(2);
  VecSet(
    hiddenLayers,
    0,
    nbIn);
  VecSet(
    hiddenLayers,
    1,
    nbIn);

  // Create the NeuraNet
  NeuraNet* nn =
    NeuraNetCreateFullyConnected(
      nbIn,
      nbOut,
      hiddenLayers);

  // Create the GradAutomatonNeuraNet
  bool diagLink = true;
  GradAutomatonNeuraNet* gann =
    GradAutomatonCreateNeuraNetSquare(
      GANNIGetDimStatus(that),
      dimImg,
      diagLink,
      nn);

  // Train the NeuraNet
  GANNITrain(
    that,
    nn,
    gann,
    gb);

  // Free memory
  GradAutomatonNeuraNetFree(&gann);
  NeuraNetFree(&nn);
  VecFree(&hiddenLayers);
  GBFree(&gb);

  // Display a message for the user
  if (GANNIGetVerbose(that) == true) {

    printf("Encoding ended\n");

  }

}

// Train the NeuraNet 'nn' of the GradAutomatonNeuraNet 'gann'
// for the GANNI 'that' on the GenBrush 'gb'
void GANNITrain(
                  GANNI* const that,
               NeuraNet* const nn,
  GradAutomatonNeuraNet* const gann,
               GenBrush* const gb) {

  // Init the random generator
  srandom(time(NULL));

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

  // Free memory
  GenAlgFree(&ga);

}

// Run the decoding process for the GANNI 'that'
void GANNIDecode(GANNI* const that) {

  (void)that;

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
