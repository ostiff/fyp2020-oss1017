# fyp2020-oss1017
Laboratory data cluster analysis

[url-documentation]: https://ostiff.github.io/fyp2020-oss1017/index.html

[Documentation][url-documentation] | Resources | Contributors | Release Notes

Sphinx documentation
--------------------
To create the documentation using html and copy it straight to
the documentation branch (gh-pages) use the following command. 

`$ make github`

In order to update the content in GitHub pages you need to commit
the changes into the repository. To track and commit all changes
use:

`$ git add -A`


Autoencoder interactive example
-------------------------------

Interactive example to visualise dengue data encoded by an auto encoder.

Features:

- Get information about the k points closest to the selected one.
- Input data corresponding to an unseen patient to get information about the k
patients which are closest in the latent space. 


With the virtual environment active:
`$ python examples/vae/vae_kdtree_server.py`

The server will be started locally on: http://127.0.0.1:5000/

The example can be accessed on http://127.0.0.1:5000/ or by opening 
`examples/vae/templates/vae_kd_tree_client.html` in a browser.
