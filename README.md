<!-- PROJECT LOGO -->


<h1 align="center">Generalizing PPO with Procgen</h1>

<div align="center">
 <figure>
    <img src="https://raw.githubusercontent.com/openai/procgen/master/screenshots/starpilot.png" alt="Star Pilot" width="150"/>
  </figure>


  <figure>
    <img src="https://raw.githubusercontent.com/openai/procgen/master/screenshots/caveflyer.png" alt="Cave Flyer" width="150"/>
  </figure>

  <figure>
    <img src="https://raw.githubusercontent.com/openai/procgen/master/screenshots/coinrun.png" alt="Coin Run" width="150"/>
  </figure>
  
 <figure>
    <img src="https://raw.githubusercontent.com/openai/procgen/master/screenshots/bigfish.png" alt="Big Fish" width="150"/>
  </figure>
</div>

<p align="center">
Procedural generation in reinforcement learning (RL) environments is a powerful
tool to test agents’ ability to generalize beyond memorized behaviors. In this
project I apply Proximal Policy Optimization (PPO), a widely used on-policy RL
algorithm, to a subset of the Procgen Benchmark, training agents in four visually
and cognitively distinct games: StarPilot, CaveFlyer, CoinRun, and BigFish. This
report details the network architectures, algorithmic decisions and the evaluation
metrics, used to assess training stability and generalization.
    <br />
    <br />
    ·
    <a href="https://github.com/alessandrablasioli/AAS-Exam-Project-Generalizing-PPO-with-Procgen/issues">Report Bug</a>
    ·
    <a href="https://github.com/alessandrablasioli/AAS-Exam-Project-Generalizing-PPO-with-Procgen/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project
Generalization in RL remains challenging, particularly when agents are trained in environments where
experiences repeat. The Procgen Benchmark addresses this by generating new levels for each
episode using procedural content generation. Sixteen such environments test both sample efficiency
and generalization, making them ideal for evaluating RL algorithms under diverse conditions.
PPO is a popular policy-gradient method thanks to its balance of sample efficiency and optimiza-
tion stability. Prior studies have demonstrated PPO’s performance on the complete Procgen suite.
However, comparative analyses across diverse games, emphasizing per-environment behavior, remain
valuable for understanding strengths and weaknesses.
This work implements and evaluates PPO in multiple Procgen environments.

The project was carried out as part of the "Autonomous and Adaptive Systems" course.


### Built With

* [TensorFlow][React-url]


### Installation
 
 Clone the repo
   ```sh
   git clone https://github.com/alessandrablasioli/AAS-Exam-Project-Generalizing-PPO-with-Procgen.git
   ```


<!-- CONTACT -->
## Contact

 [Alessandra Blasioli - LinkedIn](https://www.linkedin.com/in/alessandra-blasioli-3000531b2/) 
 
 
Project Link: [https://github.com/alessandrablasioli/AAS-Exam-Project-Generalizing-PPO-with-Procgen.git](https://github.com/alessandrablasioli/AAS-Exam-Project-Generalizing-PPO-with-Procgen.git)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[product-screenshot]: https://github.com/alessandrablasioli/JHON-AI/blob/main/img/photo5963069703016528478.jpg
[Next-url]: https://developer.android.com/studio
[React-url]: https://www.tensorflow.org/
[Vue-url]: https://www.mongodb.com/atlas/database
