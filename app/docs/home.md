# Alivio 🛖

Vulnerability Assessment Through Demographics & Building Damage Post-Natural Disasters.

![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/H3-1E54B7.svg?style=for-the-badge&logo=H3&logoColor=white)
![](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)

Code on GitHub:

[![Alivio](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/cricksmaidiene/alivio)

## Project

Disaster relief organizations face significant hurdles in efficiently identifying and assisting populations vulnerable to or affected by natural disasters. The core of the problem lies in the inability to quickly and accurately assess the potential for disaster impact and the extent of damage post-disaster. This challenge is amplified by:

* **Infrastructure & Resources**:
  * The resilience of buildings and infrastructure to withstand natural disasters is a critical factor.
  * Efficient allocation of resources, both in terms of disaster preparedness and response, is crucial.
  * Misallocation can lead to ineffective disaster management, resulting in increased casualties and property damage.

Alivio is built by utilizing the foundation computer vision model - Visual Transformer as its backbone and aims to provide a cohesive solution for rescue team member with not only identifying the building damage by severity but also identifies vulnerable populations in the affected area.

## Data

![](data:image/png;base64,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)
|![](https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg)

## Contributors

* [Yamini Gotimukul](mailto:yamini@berkeley.edu)
* [Samuel Ha](mailto:samuel@berkeley.edu)
* [Paris Li](mailto:paris@berkeley.edu)
* [Sarah Thomas](mailto:sarah@berkeley.edu)
* [Eshwaran Venkat](mailto:eshwaran@berkeley.edu)

Under the supervision of: Joyce Shen & Korin Reed, `DATASCI 210` - Capstone Spring 2024
