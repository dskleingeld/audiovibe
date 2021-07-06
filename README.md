Audiovibe
==========

I build a system that estimates the emotion or mood of a musical piece.
It performs well on the relatively small dataset but fails to generalize
as seen with a quick test. The dataset only contains 1000 songs, from
these I take only a single 45 second fragment. The small dataset is not
enough to generalize even without overfitting.

The regression problem given to the neural network can be improved,
instead of a separate valance and arousal the angle between them could
be the networks prediction. The choice for valance arousal was made
because of the availability of annotated data it might not be the right
choice for emotion prediction. Professor of psychology Lisa Feldman
Barrett feldman notes in her book on emotions:

> A careful read of the literature reveals that no theory has ever
> hypothesized that emotions can sufficiently be reduced to or explained
> by valence and arousal. Instead, these theories hypothesize that
> valence and arousal are important (and perhaps necessary) descriptive
> features of all emotions.

Going from audio features through arousal and valance to get to emotions
is throwing away information. A neural network classifying emotions
directly instead of regressing to arousal and valance is going to
perform better. Further more we should ask if we really want the emotion
of a song and not an other description. I rarely find myself looking for
music with a specific emotion. Rather I might want *energetic* and
*rough* music when feeling angry and *calm smooth* music when tired.
These are descriptions that intuitively map to certain audio features.

This seems to be the way the state of the art is developing. New far
larger datasets are now available with over 18-thousand songs annotated
with 56 moods and themes annotations including descriptions such as
*uplifting* and *meditative*. There has been a declining
interest in emotion recognition since 2014, the inclusion of theme
recognition could very well bring this around while providing more
usable results.

A more extended explanation can be found in the [report](report/report.pdf)

Run Instructions
================

Install
-------

To install you will need: make, python version 3.7 or higher and a
installation of pip for that python. Then run *make install* from the
projects root. This assumes these versions of pip and python are called
using pip3 and python3. You might need to change this in the makefile
and loosen versions in requirements.txt depending on what is available
for your installed python. To prevent messing with your current python
environment I recommend doing this in a virtual env.

Use
---

The project download comes with the trained model and annotations
database, ready to be used for emotion estimation. To estimate a songs
emotion change the variable `path` in `estemate.py` to the location of
the song for which you wish to estimate the emotion and call it using
*python3 /src/audiovibe/estimate.py*. You may wish to use
*test/cut45.sh* to cut songs to a 45 second segment (requires ffmpeg).
