# Econometrics
Graduate level econometrics notes with embedded examples using the Julia language.

To get just the notes, click on econometrics.pdf, and then on Download, at the upper R of the page, which will download only the pdf. Links in the pdf point to files here on github, and will open in your browser.

## To run the examples
You need to install this repository as a Julia package. Do this as follows:

1. git clone this repo to some convenient place on your storage media

2. add the package to Julia by doing ```] add <path where you downloaded the repo>```, for example, if you cloned it into the git directory of your home directory, you would do ```] add ~/git/Econometrics```  This will install many supporting packages, so be patient.

3. go to the location of the Econometrics code, start Julia, and do ```] activate .; instantiate```  This will install a last few packages.

4. (optional, only if you're curious) basic testing of the package has been added. Do ```using Pkg; Pkg.test("Econometrics")```

5. then do ```using Econometrics``` in Julia to use the package. You can see some examples by typing 
   ```julia
   ols()
   mleresults()
   gmmresults()
   mcmc()
   npreg()
   samin()
   ```
   

6. To see the source code for those examples, type ```edit(ols,())```, to see the OLS source code.

7. To run examples, cd into the relevant subdirectory of Econometrics/Examples, and then just include the script you would like to run.

## There are a couple of unusual thing about these notes:
- they are available in editable form (econometrics.lyx), so that you can modify them to suit your needs: see the first chapter for more information, and get LyX from  www.lyx.org. 
- they contain links that point to example programs using the Julia language. The examples show how to use the methods and illustrate properties of estimators. The example code can be modified to allow exploration.

To get an idea of how this works, the following figure shows an explanation in the pdf version of the notes, with a link to an example. The code of the example is visible in the lower R, and the output of running the example in Julia is at the lower L.

![example](https://github.com/mcreel/Econometrics/blob/master/example.png)
