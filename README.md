# Tap to Drums: Extending Monophonically Tapped Rhythms to Polyphonic Drum Generation

This repoistory contains the code used and data collected for the *Tap to Drums: Extending Monophonically Tapped Rhythms to Polyphonic Drum Generation*. Here you will find the PureData files used to run the experiment, the associated python files for the experiment and other support and initializer files.

**Abstract**
In this paper, we explore the literature surrounding rhythm perception to develop algorithms that extract a monophonic rhythm from a polyphonic drum pattern. We develop machine learning models for those algorithms to predict the pattern’s location in a polyphonic similarity based 2-D latent rhythm space. Following that we have 25 subjects tap along to polyphonic drum patterns to explore the behaviors of reproducing complex rhythms. The model was able to reasonably predict the location of a monophonic rhythm in the rhythm space (MAE=0.039, SD=0.057). Subjects tapped more accurately to an intended velocity as they became more experienced with the system. The model failed to predict the location of the subject-tapped monophonic rhythms (MAE=0.4580, SD=0.076), highlighting the need for a more thorough subject-rated investigation into refining a tap->polyphonic drums pipeline.

**Keywords**: rhythm, rhythm perception, rhythm similarity, tapping, rhythm space

Read Here:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8381068.svg)](https://doi.org/10.5281/zenodo.8381068)

To run the experiment:
1. Connect drum pad and headphones to computer. Start PureData and change Audio Output and MIDI input to respective equipment.
2. Open *tap_tests.pd*, followed by running *tap_tests.py*.
3. Test data is saved in tap-to-drums/results, subjects must manually save the first test.

![Test 1: Tap Consistency](https://github.com/peter-clark/tap-to-drums/blob/main/formatting/test_tap-consistency.png)
![Test 2: Tap Rhythm](https://github.com/peter-clark/tap-to-drums/blob/main/formatting/test_tap-rhythm.png)


To see results:
1. Open *data_analysis.py*
2. Select which approaches you want to see analysis from, set to True.
3. Run and see graphical results.

If you have any questions or comments, please contact Peter Clark (peterjosephclark1@gmail.com).

