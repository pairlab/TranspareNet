<template>
  <div id="app" >
    <div id="abs" style="dispaly: block">
    <h1>Seeing Glass: Joint Point-Cloud and Depth Completion for Transparent Objects</h1>
    <p>
      <a href="">Haoping Xu</a>, 
      <a href="">Yi Ru Wang</a>, 
      <a href="">Sagi Eppel</a>, 
      <a href="https://www.matter.toronto.edu/basic-content-page/about-alan">Alan Aspuru-Guzik</a>, 
      <a href="http://www.cs.toronto.edu/~florian/">Florian Shkurti</a>, 
      <a href="https://animesh.garg.tech/">Animesh Garg</a>
    </p>
    
    <h2> Abstract </h2>
    <p style="text-align:left;"> The basis of many object manipulation algorithms is RGB-D input. Yet,commodity RGB-D sensors can only provide distorted depth maps for a wide range of transparent objects due light refraction and absorption. To tackle the perception challenges posed by transparent objects, we propose TranspareNet, a joint point cloud and depth completion method, with the ability to complete the depth of transparent objects in cluttered and complex scenes, even with partially filled fluid contents within the vessels. To address the shortcomings of existing transparent object data collection schemes in literature, we also propose an automated dataset creation workflow that consists of robot-controlled image collection and vision-based automatic annotation. Through this automated workflow, we created Toronto Transparent Object Depth Dataset (TODD), which consists of nearly 15000 RGB-D images. Our experimental evaluation demonstrates that TranspareNet outperforms existing state-of-the-art depth completion methods on multiple datasets, including ClearGrasp, and that it also handles cluttered scenes when trained on TODD. </p>

    <div class="float-container">
        <div class="float-child"><v-btn class="ma-2" href="https://openreview.net/forum?id=tCfLLiP7vje"> Paper</v-btn> </div>
        <div class="float-child"><v-btn class="ma-2" href="https://github.com/pairlab/TranspareNet"> Code</v-btn></div>
        <div class="float-child"><v-btn class="ma-2" href="https://doi.org/10.5683/SP3/ZJJAJ3"> Dataset</v-btn></div>
    </div>
    <div class="float-container">
      <div class="float-child-half"> <h3> TranspareNet </h3> <v-img :src="`${publicPath}main.svg`" contain></v-img></div>
      <div class="float-child-half">
        <h3> Dataset collection</h3>
        <iframe width="600" height="380" src="https://www.youtube.com/embed/_HIetJ4mdlg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>
      </div>
    </div>
    <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
    <v-banner single-line>
    <h3> <p style="text-align:left;">Objects in Dataset </p>  <div id="example-5">
      
    </div></h3>
    </v-banner>
    <div class="float-container">
        <div class="float-child">
          <v-img :src="`${publicPath}cad.png`" contain></v-img>
        </div>
        <div class="float-child">
          <v-img :src="`${publicPath}glass.png`" contain></v-img>
        </div>
        <div class="float-child">
          <v-img :src="`${publicPath}filled.png`" contain></v-img>
        </div>
    </div>
<br style="clear:both" />
    <v-banner single-line>
    <h3> <p style="text-align:left;"> Dataset Capture Time-lapse </p></h3>
    <span STYLE="font-size:18.0pt" >Select Object Type:</span>
      <select v-model="objType" STYLE="font-size:18.0pt">
        <option value=0>Beaker 0</option>
        <option value="1">Beaker 1</option>
        <option value="2">Beaker 2</option>
        <option value="3">Flask 0</option>
        <option value="4">Flask 1</option>
        <option value="5">Flask 2</option>
      </select>
    </v-banner>
    <div class="float-container">
    <div id="imageGif"  class="float-child">
      <h4> RGB </h4>
      <gif-viewer width="320" height="200" v-show="objType == 0" file="image0.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 1" file="image1.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 2" file="image2.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 3 || objType == 5" file="image53.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 4" file="image24.gif"></gif-viewer>
    </div>
    <div id="depthGif"  class="float-child">
      <h4> Sensor Raw Depth </h4>
      <gif-viewer width="320" height="200" v-show="objType == 0" file="depth0.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 1" file="depth1.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 2" file="depth2.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 3 || objType == 5" file="depth53.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 4" file="depth24.gif"></gif-viewer>
    </div>
    <div id="depthGTGif"  class="float-child">
      <h4> Ground Truth Depth </h4>
      <gif-viewer width="320" height="200" v-show="objType== 0" file="depthgt0.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 1" file="depthgt1.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 2" file="depthgt2.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 3 || objType == 5" file="depthgt53.gif"></gif-viewer>
      <gif-viewer width="320" height="200" v-show="objType == 4" file="depthgt24.gif"></gif-viewer>
    </div>
    </div>
      <div class="float-container">
      <div class="float-child">
        <h4> Object CAD Model </h4>
        <model-stl :src="`${publicPath}${objType}.stl`" :height="320" :width="320" :cameraPosition=scale> </model-stl>
      </div>
      <div class="float-child">
        <h4> Raw Depth Point Cloud</h4>
        <model-ply  :src="`${publicPath}depth2pcd_${objType}.ply`" :height="320" :width="320" > </model-ply>
        <!-- <model-obj  :src="`${publicPath}depth2pcd_${objType}.obj`" :height="600" :width="600" > </model-obj> -->
      </div>
      <div class="float-child">
        <h4> Ground Truth Depth Point Cloud</h4>
        <model-ply  :src="`${publicPath}depth2pcd_GT_${objType}.ply`" :height="320" :width="320" > </model-ply>
        <!-- <model-obj  :src="`${publicPath}depth2pcd_GT_${objType}.obj`" :height="600" :width="600" > </model-obj> -->
      </div>
    </div>
    <div class="float-container">
      <h2> <p style="text-align:left;">Acknowledgements</p></h2>
      <p style="text-align:left;">This Project is supported by the following.</p>
      <div class="float-mini"><v-img :src="`${publicPath}toronto.jpg`" contain></v-img></div>
      <div class="float-mini"><v-img :src="`${publicPath}vector.jpg`" contain></v-img></div> 
      <div class="float-minih"><v-img :src="`${publicPath}pair.png`" contain></v-img></div> 
      <div class="float-minih"><v-img :src="`${publicPath}matter.svg`" contain></v-img></div> 
    </div> 
  </div>

  
</template>

<script>
import GifViewer from './components/GifViewer.vue';
import { ModelStl, ModelPly} from 'vue-3d-model';
export default {
  name: 'App',
  components: {
    GifViewer,
    ModelStl,
    ModelPly
  },
  data (){
    return {
      objType: 0,
      items:[{name:"Beaker 0", index:0}, {name:"Beaker 1", index:1}, {name:"Beaker 2", index:2},{name:"Flask 0", index:3}, {name:"Flask 1", index:4}, {name:"Falsk 2", index:5}] ,
      publicPath: process.env.BASE_URL,
      scale: { x: 200, y: 0, z: -3 },
      videoId: "https://www.youtube.com/watch?v=mfL8tZUKRW4",
      playerVars: {autoplay: 1}
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
  margin-left: auto;
  margin-right: auto;
  max-width: 80rem;
}
#abs {
  margin-left: auto;
  margin-right: auto;
  margin-bottom: auto;
  display: block;
}
p {
      margin-bottom: 1rem;
}
.float-container {
    padding: 0px;
    display: block;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.container {
    height: 100%;
    width: 100%;
}
.float-child {
    width: 33%;
    float: left;
}  
.float-child-half {
    width: 50%;
    height: 400px;
    float: left;
} 
.float-mini {
    width: 5%;
    float: left;
    margin-left: 2rem;
    margin-right: 2rem;
}
.float-minih {
    width: 10%;
    float: left;
    margin-left: 2rem;
    margin-right: 2rem;
}
.triangle {
    width: 0;
    height: 0;
    border: solid 30px;
    margin: 20px;
}
.top {
    border-color: transparent transparent red transparent;
}
.left {
    border-color: transparent transparent transparent red;
}
.bottom {
    border-color: red transparent transparent transparent;
}
.right {
    border-color: transparent red transparent transparent;
}


</style>
