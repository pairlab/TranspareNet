<template>
  <div id="app" style="background-color:#EFEFEF;" >
    <div id="abs">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<div class="topnav" style="text-align:right" id="myTopnav">
  <a href="#home" class="active">5th Conference on Robot Learning (CoRL 2021)</a>
  <a href="#Architecture">Architecture</a>
  <a href="#Collection">Dataset Collection</a>
  <a href="#Dataset">Dataset</a>
  <a href="#Explorer">Dataset Explorer</a>
  <a href="#Acknowledgements">Acknowledgements</a>
</div>
    <h1 style="color:#4484CE; margin-left: 70px; margin-top: 40px; margin-right: 70px;font-size:2.5vw"> Seeing Glass: Joint Point-Cloud and Depth Completion for Transparent Objects</h1>
    <p style="color:#565656;">
      <a style="color:#565656;font-size:1.5vw" href="https://www.linkedin.com/in/haoping-xu-0b3b44a4/">Haoping Xu</a><sup>* 1,2</sup>, 
      <a style="color:#565656;font-size:1.5vw" href="https://www.linkedin.com/in/yi-ru-helen-wang/">Yi Ru Wang</a><sup>* 1,2</sup>, 
      <a style="color:#565656;font-size:1.5vw" href="">Sagi Eppel</a><sup>1,2</sup>, 
      <a style="color:#565656;font-size:1.5vw" href="https://www.matter.toronto.edu/basic-content-page/about-alan">Alan Aspuru-Guzik</a><sup>1,2</sup>, 
      <a style="color:#565656;font-size:1.5vw" href="http://www.cs.toronto.edu/~florian/">Florian Shkurti</a><sup>1,2</sup>, 
      <a style="color:#565656;font-size:1.5vw" href="https://animesh.garg.tech/">Animesh Garg</a><sup>1,2,3</sup>
    </p>
    
    <div class="col" style="font-size:1vw">
                <p><sup>1</sup>University of Toronto &nbsp;
                <sup>2</sup>Vector Institute &nbsp;
                <sup>3</sup>NVIDIA &nbsp;
                <sup>*</sup>Equal Contribution &nbsp; </p>
            </div>
    <h2 id = "Abstract"> Abstract </h2>
    <p style="text-align:left;margin-left: 40px; margin-right: 40px;"> The basis of many object manipulation algorithms is RGB-D input. Yet,commodity RGB-D sensors can only provide distorted depth maps for a wide range of transparent objects due light refraction and absorption. To tackle the perception challenges posed by transparent objects, we propose TranspareNet, a joint point cloud and depth completion method, with the ability to complete the depth of transparent objects in cluttered and complex scenes, even with partially filled fluid contents within the vessels. To address the shortcomings of existing transparent object data collection schemes in literature, we also propose an automated dataset creation workflow that consists of robot-controlled image collection and vision-based automatic annotation. Through this automated workflow, we created Toronto Transparent Object Depth Dataset (TODD), which consists of nearly 15000 RGB-D images. Our experimental evaluation demonstrates that TranspareNet outperforms existing state-of-the-art depth completion methods on multiple datasets, including ClearGrasp, and that it also handles cluttered scenes when trained on TODD. </p>

    <div class="float-container">
        <div class="float-child"><v-btn class="ma-2" href="https://openreview.net/forum?id=tCfLLiP7vje"> Paper</v-btn> </div>
        <div class="float-child"><v-btn class="ma-2" href="https://github.com/pairlab/TranspareNet"> Code</v-btn></div>
        <div class="float-child"><v-btn class="ma-2" href="https://doi.org/10.5683/SP3/ZJJAJ3"> Dataset</v-btn></div>
    </div>
    <div style="margin-top:6rem; margin-bottom:6rem">
    <iframe width="640" height="480" src="https://www.youtube.com/embed/ELzNzE48FFY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
    <div class="float-container" style="margin-left: 70px; margin-right: 70px;">
      <div class="float-child-half"> <h2 id = "Architecture"> TranspareNet </h2> <v-img :src="`${publicPath}main.svg`" width="500" height="300" contain></v-img>
      <p STYLE="text-align:left;font-size:10.0pt;margin-right:40px;"> Transparent object depth is de-projected to a point cloud, and put through Point Cloud Completion module to get the final point cloud. The final point cloud is then projected to the depth domain and replaces the original depth of the transparent object within the mask. An encoder-decoder based Depth Completion module takes combined depth and RGB signal as input, and the decoder modulation branch takes the object mask as input for modulation. Finally, the decoder outputs the predicted completed depth.</p>
      </div>
      <div class="float-child-half">
        <h2 id = "Collection"> Automated Dataset collection </h2>
        <iframe width="500" height="300" src="https://www.youtube.com/embed/_HIetJ4mdlg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <p STYLE="text-align:left;font-size:10.0pt;margin-left:20px;margin-right:80px;"> A commodity RGB-D sensor is mounted to the robot arm's end effector. The scene with the transparent object is scanned from multiple viewing angles to collect the raw depth. AprilTags on the base template are detected for each image, and based on their 6DoF poses and the known translation between tags and objects, we can fit the 3D model of object(s) to their respective locations. The result of this automatic collection and annotation process is the RGB image, instance object segmentation, raw depth, ground truth depth.</p>
      </div>

    </div>
    </div>
    <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
    <h2 id = "Dataset"> Toronto Transparent Objects Depth Dataset (TODD) </h2>
    <p style="text-align:left;margin-left: 40px; margin-right: 40px;"> In total, Toronto Transparent Objects Depth Dataset (TODD) has 14,659 images of scenes which contains six glass beakers and flasks in five different backgrounds. Four objects are used in training set and the other two novel objects form the novel validation and test set. The training set has 10,302 images where validation and testing set combined has 4357 images. Every scene consists of up to three transparent objects with occlusion, which introduces additional complexity to the dataset. The objects and their placement are selected to mimic real-life transparent glassware, which can help to develop vision aware robots capable of manipulating transparent vessels.</p>

    <div class="float-container" style="margin-left: 70px; margin-right: 70px;">

        <div class="float-child">
          <v-img :src="`${publicPath}cad.png` " max-width="300" max-height="200" contain></v-img>
        </div>
        <div class="float-child">
          <v-img :src="`${publicPath}glass.png`" max-width="300" max-height="200" contain></v-img>
        </div>
        <div class="float-child">
          <v-img :src="`${publicPath}filled.png`" max-width="300" max-height="200" contain></v-img>
        </div>
    </div>
<br style="clear:both" />
    <v-banner single-line>
    <h3 id = "Explorer"> Dataset Explorer </h3>
    <span STYLE="font-size:12.0pt" >Select Object Type:   </span>
      <select v-model="objType" STYLE="font-size:12.0pt; background-color:#4484CE; border-radius: 10px;">
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
    <div class="float-container" style="text-align:center">
      <h2 id="Acknowledgements"> Acknowledgements </h2>
      
      <p style="text-align:left;margin-left: 40px; margin-right:40px">Animesh Garg is a CIFAR AI Chair. Alan Aspuru-Guzik is a CIFAR AI Chair and CIFAR Lebovic Fellow. Animesh Garg and Florian Shkurti are also supported in part through the NSERC Discovery Grants Program. The authors would like to acknowledge Vector Institute and ComputeCanada for computing services. Alan Aspuru-Guzik and Haoping Xu thank the Canada 150 Research Chair funding from NSERC, Canada. Alan Aspuru-Guzik is thankful for the generous support of Dr. Anders G. Froseth. The authors would like to thank Yuchi Zhao for constructive feedback and discussions on the manuscript.</p>
      <center>
      <div class="float-mini"><v-img :src="`${publicPath}toronto.jpg`" contain></v-img></div>
      <div class="float-mini"><v-img :src="`${publicPath}vector.jpg`" contain></v-img></div> 
      <div class="float-minih"><v-img :src="`${publicPath}pair.png`" contain></v-img></div> 
      <div class="float-minih"><v-img :src="`${publicPath}matter.svg`" contain></v-img></div> 
      </center>
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
  margin-top: 0px;
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
    margin-left: 40px;
    margin-right:40px;
    margin-top: 2rem;
    margin-bottom: 2rem;
    height: 100%;
    width: 100%;
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
h2   {color: #ff8800;font-size:1.5vw}
p    {color: #565656;font-size:1vw}

/* Add a black background color to the top navigation */
.topnav {
  background-color: #4484CE;
  overflow: hidden;
  position: sticky;
}

/* Style the links inside the navigation bar */
.topnav a {
  float: left;
  color: #f2f2f2;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
  font-size: 17px;
}

/* Change the color of links on hover */
.topnav a:hover {
  background-color: #ddd;
  color: black;
}

/* Add a color to the active/current link */
.topnav a.active {
  background-color: #ff8800;;
  color: white;
}
ul {
  white-space: nowrap;
}

ul, li {
  list-style: none;
  display: inline;
}
select{
  width: 80px;
  height: 27px;
  padding: 5px;
  color: white;
  background-color: #4484CE;
  text-align:center
}
select option { color: #ff8800; }
select option:first-child{
  color: green;
}

</style>
