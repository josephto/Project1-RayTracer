// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

using namespace glm;

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	vec3 jitter = 2.0f*generateRandomNumberFromThread(resolution, time, x, y);
	float NDCx = ((float)x +jitter.x)/resolution.x;
	float NDCy = ((float)y +jitter.y )/resolution.y;
	
	//float NDCx = ((float)x )/resolution.x;
	//float NDCy = ((float)y )/resolution.y;

	vec3 A = cross(view, up);
	vec3 B = cross(A, view);

	vec3 M = eye+view;
	vec3 V = B * (1.0f/length(B)) * length(view)*tan(radians(fov.y));
	vec3 H = A * (1.0f/length(A)) * length(view)*tan(radians(fov.x));

	vec3 point = M + (2*NDCx -1)*H + (1-2*NDCy)*V;

	ray r;
	r.origin = eye;
	r.direction = normalize(point-eye);
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x/iterations*255.0;
      color.y = image[index].y/iterations*255.0;
      color.z = image[index].z/iterations*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, 
							int numberOfCubes, int numberOfSpheres, material* cudamaterials, int numberOfMaterials, int* cudalights, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){

		ray rayFromCamera = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);

		//find aim point
		vec3 aimPoint = rayFromCamera.origin + cam.focalLength*rayFromCamera.direction;

		//jittered ray (DOF)
		float degOfJitter = 1;
		vec3 jitter = generateRandomNumberFromThread(resolution, time, x, y);
		ray jitteredRay;
		jitteredRay.origin = vec3(rayFromCamera.origin.x+degOfJitter*jitter.x, rayFromCamera.origin.y+degOfJitter*jitter.y, rayFromCamera.origin.z);	
		jitteredRay.direction = normalize(aimPoint-jitteredRay.origin);

		ray currentRay = jitteredRay;
		
		int rayCount = 0;
		vec3 realColor = vec3(0,0,0);
		vec3 accumulColor = vec3(0,0,0);
		vec3 accumulReflectiveSurfaceColor = vec3(1,1,1);

		while(rayCount <= rayDepth){
			float tempLength, closest = 1e26;
			int closestObjectid;
			vec3 tempIntersectionPoint, tempNormal, normal, intersectionPoint;
			vec3 pixelColor, objectColor, specColor;
			float specExponent, isReflective;

			for (int i = 0; i < numberOfCubes; i++){
				if(geoms[i].type == CUBE){
					tempLength = boxIntersectionTest( geoms[i], currentRay, tempIntersectionPoint, tempNormal);
				}

				if (tempLength < closest && tempLength >= 0){
					closest = tempLength;
					normal = tempNormal;
					intersectionPoint = tempIntersectionPoint;
					closestObjectid = i;
				}
			}

			for(int i = numberOfCubes; i < numberOfGeoms; i++){
				if(geoms[i].type == SPHERE){
					tempLength = sphereIntersectionTest( geoms[i], currentRay, tempIntersectionPoint, tempNormal);
				}

				if (tempLength < closest && tempLength >= 0){
					closest = tempLength;
					normal = tempNormal;
					intersectionPoint = tempIntersectionPoint;
					closestObjectid = i;
				}
			}

			pixelColor = vec3(0,0,0);

			if (closest < 1e26 && closest >= 0){

				objectColor = cudamaterials[geoms[closestObjectid].materialid].color;
				specExponent = cudamaterials[geoms[closestObjectid].materialid].specularExponent;
				specColor = cudamaterials[geoms[closestObjectid].materialid].specularColor;
				isReflective = cudamaterials[geoms[closestObjectid].materialid].hasReflective;

				vec3 accumulDiffuse = vec3(0,0,0);
				vec3 accumulSpec = vec3(0,0,0);
				vec3 ambient = objectColor;
				vec3 reflectedDir;
			
				for (int j = 0; j < numberOfLights; j++){
					if (closestObjectid == cudalights[j]){
						pixelColor = cudamaterials[geoms[closestObjectid].materialid].color;
						colors[index] += pixelColor;
						return;
					}

					vec3 randomPointOnLight;
					if (geoms[cudalights[j]].type == CUBE)
						randomPointOnLight = getRandomPointOnCube(geoms[cudalights[j]],time);
					else if (geoms[cudalights[j]].type == SPHERE)
						randomPointOnLight = getRandomPointOnSphere(geoms[cudalights[j]],time);


					vec3 lightDir = normalize(randomPointOnLight - intersectionPoint);

					//ambient

			
					//diffuse
					vec3 diffuse = dot(normal, lightDir) * cudamaterials[geoms[cudalights[j]].materialid].color * (objectColor);
					diffuse = vec3(clamp(diffuse.x, 0.0, 1.0), clamp(diffuse.y, 0.0, 1.0), clamp(diffuse.z, 0.0, 1.0));


					vec3 specular = vec3(0,0,0);
					reflectedDir = currentRay.direction - vec3(2*vec4(normal*(dot(currentRay.direction,normal)),0));
					reflectedDir = normalize(reflectedDir);
					//specular phong lighting
					if (specExponent > 0){
						//vec3 reflectedDir = lightDir -  vec3(2*vec4(normal*(dot(lightDir,normal)),0));
						float D = dot(reflectedDir, lightDir);
						if (D < 0) D = 0;
						specular = specColor*pow(D, specExponent);
					}

					//shadows, see if there is an object between light and pixel.
					ray pointToLight; 
					pointToLight.origin = intersectionPoint;
					pointToLight.direction = lightDir;
					float lengthFromPointToLight;
					if (geoms[cudalights[j]].type == CUBE)
						lengthFromPointToLight = boxIntersectionTest( geoms[cudalights[j]], pointToLight, tempIntersectionPoint, tempNormal);
					else if (geoms[cudalights[j]].type == SPHERE)
						lengthFromPointToLight = sphereIntersectionTest( geoms[cudalights[j]], pointToLight, tempIntersectionPoint, tempNormal);
					tempLength = 1e26;
					int occluded = -1;
					for (int i = 0; i < numberOfGeoms; i++){
						if (i != closestObjectid){
							if(geoms[i].type == CUBE){
								tempLength = boxIntersectionTest( geoms[i], pointToLight, tempIntersectionPoint, tempNormal);
							}else{
								tempLength = sphereIntersectionTest(geoms[i], pointToLight, tempIntersectionPoint, tempNormal);
							}

							if (tempLength < lengthFromPointToLight && tempLength != -1){
								occluded = i;
								i = numberOfGeoms; 
							}
						}
					}

					//apply shadow, make darker
					bool hitLight = false;
					for (int x = 0; x < numberOfLights; x++){
						if (occluded == cudalights[x]){
							hitLight = true;
							break;
						}
					}
					if (occluded != -1 && !hitLight){
						//diffuse *= .1f;
						//specular *= .1f;
						diffuse = vec3(0,0,0);
						specular = vec3(0,0,0);
					}

					accumulDiffuse += diffuse;
					accumulDiffuse = clamp(accumulDiffuse, vec3(0,0,0), objectColor);
					accumulSpec += specular;
					accumulSpec = clamp(accumulSpec, vec3(0,0,0), vec3(1,1,1));

				}//for loop

				if (specExponent > 0){
					accumulColor += .4f*accumulDiffuse + .5f*accumulSpec + .1f*ambient;
				}else{
					accumulColor += accumulDiffuse;
				}
				accumulColor = clamp(accumulColor, vec3(0,0,0), vec3(1,1,1));

				if (isReflective){
					currentRay.origin = intersectionPoint+0.0001f*reflectedDir;
					currentRay.direction = reflectedDir;
					accumulReflectiveSurfaceColor *= accumulColor;
				}else{
					rayCount = rayDepth;
				}


			}//if intersects with anything

			rayCount++;
			//realColor += accumulColor;
			//realColor = clamp(realColor, vec3(0,0,0), vec3(1,1,1));

		}//while loop

		realColor = accumulReflectiveSurfaceColor*accumulColor;
		colors[index] += realColor;
   }
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, int numberOfCubes, int numberOfSpheres, bool cameraMoved){
  
  int traceDepth = 5; //determines how many bounces the raytracer traces
  std::vector<int> lightsid;

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
	if (materials[newStaticGeom.materialid].emittance > 0)
		lightsid.push_back(i);
  }

  int* lightsList = new int[lightsid.size()];
  for (int i = 0; i < lightsid.size(); i++){
	  lightsList[i] = lightsid[i];
  }

     
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  int* cudalights = NULL;
  cudaMalloc((void**)&cudalights, lightsid.size()*sizeof(int));
  cudaMemcpy( cudalights, lightsList, lightsid.size()*sizeof(int), cudaMemcpyHostToDevice);
    
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.focalLength = renderCam->focalLengths[frame];

  //clear image
  if (cameraMoved)
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cudaimage);

  if (numberOfGeoms != numberOfCubes+numberOfSpheres){
	  std::cout<<"ERROR numberOfGeoms != numberOfCubes+numberOfSpheres"<<std::endl;
	  std::cout<<numberOfGeoms<<", "<<numberOfCubes<<", "<<numberOfSpheres<<std::endl;
  }

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, 
													cudaimage, cudageoms, numberOfGeoms, numberOfCubes, numberOfSpheres, cudamaterials, numberOfMaterials, cudalights, lightsid.size());

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();
  checkCUDAError("Kernel failed!");
}
