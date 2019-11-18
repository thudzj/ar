<!DOCTYPE html>
<html lang="en">
    <head>
        <!-- Required meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css" integrity="sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M" crossorigin="anonymous">

        <link href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.8.0/github-markdown.min.css" rel="stylesheet" type="text/css" />

        <link href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/styles/github.min.css" rel="stylesheet" type="text/css" />
        <link href="https://use.fontawesome.com/releases/v5.0.6/css/all.css" rel="stylesheet">
<script data-ad-client="ca-pub-3773134145653736" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
        <link href="/static/css/style.css" rel="stylesheet" type="text/css" />
    </head>
    <body>
        <div class="main">
            <div class="container-fluid">
                <div class="paths">
                    
                    
                    <span class="path"><a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a">Root</a></span>
                
                    
                    <span class="path"><a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training">gan_training</a></span>
                
                    
                    <span class="path"><a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models">models</a></span>
                
                    
                    <span class="path"><a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/resnet2.py">resnet2.py</a></span>
                
                </div>
                <div class="files">
                    
                    <div class="blob ">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/__init__.py">
                            __init__.py
                        </a>
                    </div>
                    
                    <div class="blob ">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/mlp.py">
                            mlp.py
                        </a>
                    </div>
                    
                    <div class="blob ">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/resnet.py">
                            resnet.py
                        </a>
                    </div>
                    
                    <div class="blob ">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/resnet1.py">
                            resnet1.py
                        </a>
                    </div>
                    
                    <div class="blob active">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/resnet2.py">
                            resnet2.py
                        </a>
                    </div>
                    
                    <div class="blob ">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/resnet3.py">
                            resnet3.py
                        </a>
                    </div>
                    
                    <div class="blob ">
                        <a href="/repository/a5e02628-d2d6-43b4-b645-5752fb87637a/gan_training/models/resnet4.py">
                            resnet4.py
                        </a>
                    </div>
                    
               </div>
                
                <pre><code>import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        ny = nlabels

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_0_1 = ResnetBlock(1*nf, 2*nf)

        self.resnet_1_0 = ResnetBlock(2*nf, 2*nf)
        self.resnet_1_1 = ResnetBlock(2*nf, 4*nf)

        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_2_1 = ResnetBlock(4*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 8*nf)
        self.resnet_3_1 = ResnetBlock(8*nf, 16*nf)

        self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_4_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_5_1 = ResnetBlock(16*nf, 16*nf)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)


    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
</code></pre>
                
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.11.0/umd/popper.min.js" integrity="sha384-b/U6ypiBEHpOf/4+1nzFpr53nxSS+GLCkfwBdFNTxtclqqenISfwAzpKaMNFNmj4" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/js/bootstrap.min.js" integrity="sha384-h0AbiXch4ZDo7tp9hKZ4TsHbi047NrKGLO3SEJAg45jXxnGIfYzk4Si90RDIqNm1" crossorigin="anonymous"></script>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/highlight.min.js" crossorigin="anonymous"></script>
        <script>hljs.initHighlightingOnLoad();</script>
        <script>
        (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
})(window,document,'script','//ana.durieux.me/analytics.js','ga');
ga('provide', 'adblockTracker', function(tracker, opts) {

  var xhr = new XMLHttpRequest(),
      method = "GET",
      url = "//ana.durieux.me/advertisement.js";
  try {
    xhr.open(method, url, false);
    xhr.send();
    ga('set', 'dimension' + opts.dimensionIndex, xhr.responseText != "var canRunAds=true;");
  } catch {
    ga('set', 'dimension' + opts.dimensionIndex, xhr.responseText != "var canRunAds=true;");
  }
});
ga('create', 'UA-5954162-28', 'auto');
ga('require', 'adblockTracker', {dimensionIndex: 1});
ga('send', 'pageview');
        </script>
    </body>
</html>