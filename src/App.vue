<template>
  <div id="app" >
    <div>
    <h1>Seeing Glass: Joint Point-Cloud and Depth Completion for Transparent Objects</h1>
    <p>
      <a href="">Haoping Xu</a>, 
      <a href="">Yi Ru Wang</a>, 
      <a href="">Sagi Eppel</a>, 
      <a href="https://www.matter.toronto.edu/basic-content-page/about-alan">Alan Aspuru-Guzik</a>, 
      <a href="http://www.cs.toronto.edu/~florian/">Florian Shkurti</a>, 
      <a href="https://animesh.garg.tech/">Animesh Garg</a>
    </p>
    <p style="text-align:left">
      <em>
        <a href="https://openreview.net/forum?id=tCfLLiP7vje"> CoRL 2021 Paper</a>
      </em>
    </p>
    <h2> Abstract </h2>
    <p style="text-align:left;"> The basis of many object manipulation algorithms is RGB-D input. Yet,commodity RGB-D sensors can only provide distorted depth maps for a wide range of transparent objects due light refraction and absorption. To tackle the perception challenges posed by transparent objects, we propose TranspareNet, a joint point cloud and depth completion method, with the ability to complete the depth of transparent objects in cluttered and complex scenes, even with partially filled fluid contents within the vessels. To address the shortcomings of existing transparent object data collection schemes in literature, we also propose an automated dataset creation workflow that consists of robot-controlled image collection and vision-based automatic annotation. Through this automated workflow, we created Transparent Object Depth Dataset (TODD), which consists of nearly 15000 RGB-D images. Our experimental evaluation demonstrates that TranspareNet outperforms existing state-of-the-art depth completion methods on multiple datasets, including ClearGrasp, and that it also handles cluttered scenes when trained on TODD. </p>
    </div>
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
    <span>Object Type:</span>
      <select v-model="objType">
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
      <gif-viewer v-show="objType == 0" file="image0.gif"></gif-viewer>
      <gif-viewer v-show="objType == 1" file="image1.gif"></gif-viewer>
      <gif-viewer  v-show="objType == 2" file="image2.gif"></gif-viewer>
      <gif-viewer v-show="objType == 3 || objType == 5" file="image53.gif"></gif-viewer>
      <gif-viewer v-show="objType == 4" file="image24.gif"></gif-viewer>
    </div>
    <div id="depthGif"  class="float-child">
      <h4> Sensor Raw Depth </h4>
      <gif-viewer v-show="objType == 0" file="depth0.gif"></gif-viewer>
      <gif-viewer v-show="objType == 1" file="depth1.gif"></gif-viewer>
      <gif-viewer v-show="objType == 2" file="depth2.gif"></gif-viewer>
      <gif-viewer v-show="objType == 3 || objType == 5" file="depth53.gif"></gif-viewer>
      <gif-viewer v-show="objType == 4" file="depth24.gif"></gif-viewer>
    </div>
    <div id="depthGTGif"  class="float-child">
      <h4> Ground Truth Depth </h4>
      <gif-viewer v-show="objType== 0" file="depthgt0.gif"></gif-viewer>
      <gif-viewer v-show="objType == 1" file="depthgt1.gif"></gif-viewer>
      <gif-viewer v-show="objType == 2" file="depthgt2.gif"></gif-viewer>
      <gif-viewer v-show="objType == 3 || objType == 5" file="depthgt53.gif"></gif-viewer>
      <gif-viewer v-show="objType == 4" file="depthgt24.gif"></gif-viewer>
    </div>
    </div>
      <div class="float-container">
      <div class="float-child">
        <h4> Object CAD Model </h4>
        <model-stl :src="`${publicPath}${objType}.stl`" :height="600" :width="600" :cameraPosition=scale> </model-stl>
      </div>
      <div class="float-child">
        <h4> Raw Depth Point Cloud</h4>
        <model-ply  :src="`${publicPath}depth2pcd_${objType}.ply`" :height="600" :width="600" > </model-ply>
        <!-- <model-obj  :src="`${publicPath}depth2pcd_${objType}.obj`" :height="600" :width="600" > </model-obj> -->
      </div>
      <div class="float-child">
        <h4> Ground Truth Depth Point Cloud</h4>
        <model-ply  :src="`${publicPath}depth2pcd_GT_${objType}.ply`" :height="600" :width="600" > </model-ply>
        <!-- <model-obj  :src="`${publicPath}depth2pcd_GT_${objType}.obj`" :height="600" :width="600" > </model-obj> -->
      </div>
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
  max-width: 42rem;
  margin-left: auto;
  margin-right: auto;
}
p {
      margin-bottom: 1rem;
}
.float-container {
    padding: 0px;
}
.container {
    height: 100%;
    width: 100%;
}
.float-child {
    width: 100%;
    float: left;
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
