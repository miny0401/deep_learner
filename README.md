# deep learner
deep learner make you switch tensorflow and pytorch flexiably.

## 介绍
如果只有一个深度学习框架可用，我想我就不用纠结了，从最开始使用tensorflow，到接触到pytorch后，感受到了pytorch的真香定律，但在公司里使用pytorch主要是研究使用，真要落地应用，还真是很不方便，而随着tensorflow2.0的到来，我意识到了，我写的用于pytorch小框架再进行一定的适配后，直接就可以套在tensorflow上了，也就是说，对于一个pytorch应用，不需要太多的修改就可以迁移到tensorflow2.0了！

* 框架的结构是参考了fastai的源码写的，非常的简单，方便维护和更改，
* progress.py也是参考fastai的fastprogress写的，由于有一点点不同，所以直接把源码拿过来修改应用了，由于参考了fastai的progress，所以在jupyter运行时的输出会更美观
* torch使用learner/torch_learner.py，tensorflow使用learner/tf_learner.py，使用方式非常相似
* torch和tensorflow的dataset需要自己写过
* torch和tensorflow的模型可以做到非常相似，切换改动很小
* torch和tensorlfow的自定义loss也可以做到非常相似，切换改动很小
* 查看mnist_cls里的torch_minst和tf_minst代码，可以发现非常的相似
* 框架肯定还有不足，但以后会不断完善，关键的是，由于很简单，方便自己以后维护和定制化修改

## 测试
1. 执行zsh_env_setup.sh或bash_env_setup.sh脚本，设置python的环境变量，使可访问到leaner下的脚本
2. 训练并测试 torch mnist
    ```sh
    python mnist_cls/torch_mnist.py
    python mnist_cls/torch_mnist_visualize.py
    python mnist_cls/torch_mnist_visualize_center_loss.py
    jupyter notebook # 运行torch_mnist_cls.ipynb
    ```
3. 训练并测试 tensorflow mnist
    ```sh
    python mnist_cls/tf_mnist.py
    python mnist_cls/tf_mnist_visualize.py
    python mnist_cls/tf_mnist_visualize_center_loss.py
    jupyter notebook # 运行tf_mnist_cls.ipynb
    ```
