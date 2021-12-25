\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pgffigure}[2015/01/09]

\newif\ifpgffigurefinal

\newif\ifpgffigurecyr

\RequirePackage{xkeyval}

\DeclareOption{final}{%
	\PassOptionsToClass{crop=true}{standalone}%
	\pgffigurefinaltrue%
}
\DeclareOption{nofinal}{%
	%\LoadClassWithOptions[tikz,convert=true,crop=false]{standalone}%
	\PassOptionsToClass{crop=false}{standalone}%
	\pgffigurefinalfalse%
}

\DeclareOption{cyr}{%
	\AtEndOfClass{\RequirePackage{icomma}}%
	\pgffigurecyrtrue%
}
\DeclareOption{nocyr}{%
	\pgffigurecyrfalse%
}

\newcommand{\septhouthand}{}
%\DeclareOptionX{sep1000}[{,}]{\renewcommand{\septhouthand}{#1}}


\ExecuteOptions{final}
\ExecuteOptions{nocyr}
%\ExecuteOptionsX{sep1000={{,}}}

\ProcessOptionsX

\ProcessOptions\relax


\LoadClass[tikz,convert=true]{standalone}



\RequirePackage{pgf}
\RequirePackage{pgfplots}
\RequirePackage{bm}
\RequirePackage{mathrsfs}
\RequirePackage{amsmath}
\RequirePackage{amstext}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{eucal}
\RequirePackage{upgreek}
\RequirePackage{bigints}
\RequirePackage{amsfonts}
%\define@key{}{sep1000}[]{\renewcommand{\septhouthand}{#1}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\makeatletter
\def\tikz@Get@x@coor#1,#2\tikz@sentinel{%
	\pgfmathparse{int(round(#1/1cm))}}
\def\tikz@Get@y@coor#1,#2\tikz@sentinel{%
	\pgfmathparse{int(round(#2/1cm))}}
\tikzset{%
	show grid integer coordinates/.style = {%
		preaction = {%
			decoration = {%
				show path construction,
				moveto code = {%
					\expandafter\tikz@Get@x@coor\tikzinputsegmentfirst\tikz@sentinel
					\edef\tikz@tempa{\pgfmathresult}%
					\expandafter\tikz@Get@x@coor\tikzinputsegmentlast\tikz@sentinel
					\edef\tikz@tempb{\pgfmathresult}%
					\pgfmathparse{equal(\tikz@tempa,\tikz@tempb)}%
					\ifnum\pgfmathresult=1\relax
					\node[anchor=north] at (\tikzinputsegmentfirst) {%
						\tikz@tempa};
					\draw[help lines,step=.1] (\tikzinputsegmentfirst) grid
					($(\tikzinputsegmentlast)+(1,0)$);
					\else
					\node[anchor=east] at (\tikzinputsegmentfirst) {%
						\expandafter\tikz@Get@y@coor\tikzinputsegmentfirst\tikz@sentinel\pgfmathresult};
					\fi}},
			decorate}}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@showgrid@grid
\newif\if@showgrid@left
\newif\if@showgrid@right
\newif\if@showgrid@below
\newif\if@showgrid@above
\tikzset{%
	every show grid/.style={},
	show grid/.style={execute at end picture={\@showgrid{grid=true,#1}}},%
	show grid/.default={true},
	show grid/.cd,
	labels/.style={font={\sffamily\small},help lines},
	xlabels/.style={},
	ylabels/.style={},
	keep bb/.code={\useasboundingbox (current bounding box.south west) rectangle (current bounding box.north west);},
	true/.style={left,below},
	false/.style={left=false,right=false,above=false,below=false,grid=false},
	none/.style={left=false,right=false,above=false,below=false},
	all/.style={left=true,right=true,above=true,below=true},
	grid/.is if=@showgrid@grid,
	left/.is if=@showgrid@left,
	right/.is if=@showgrid@right,
	below/.is if=@showgrid@below,
	above/.is if=@showgrid@above,
	false,
}
%    /pgf/number format/fixed zerofill, 

\def\@showgrid#1{%
	\begin{scope}[every show grid,show grid/.cd,#1]
		\if@showgrid@grid
		\begin{pgfonlayer}{background}
			\draw [help lines]
			(current bounding box.south west) grid
			(current bounding box.north east);
			%
			\pgfpointxy{1}{1}%
			\edef\xs{\the\pgf@x}%
			\edef\ys{\the\pgf@y}%
			\pgfpointanchor{current bounding box}{south west}
			\edef\xa{\the\pgf@x}%
			\edef\ya{\the\pgf@y}%
			\pgfpointanchor{current bounding box}{north east}
			\edef\xb{\the\pgf@x}%
			\edef\yb{\the\pgf@y}%
			\pgfmathtruncatemacro\xbeg{ceil(\xa/\xs)}
			\pgfmathtruncatemacro\xend{floor(\xb/\xs)}
			\if@showgrid@below
			\foreach \X in {\xbeg,...,\xend} {
				\node [below,show grid/labels,show grid/xlabels] at (\X,\ya) {\X};
			}
			\fi
			\if@showgrid@above
			\foreach \X in {\xbeg,...,\xend} {
				\node [above,show grid/labels,show grid/xlabels] at (\X,\yb) {\X};
			}
			\fi
			\pgfmathtruncatemacro\ybeg{ceil(\ya/\ys)}
			\pgfmathtruncatemacro\yend{floor(\yb/\ys)}
			\if@showgrid@left
			\foreach \Y in {\ybeg,...,\yend} {
				\node [left,show grid/labels,show grid/ylabels] at (\xa,\Y) {\Y};
			}
			\fi
			\if@showgrid@right
			\foreach \Y in {\ybeg,...,\yend} {
				\node [right,show grid/labels,show grid/ylabels] at (\xb,\Y) {\Y};
			}
			\fi
		\end{pgfonlayer}
		\fi
	\end{scope}
}
\makeatother
%\tikzset{showgrid} % would enable it globally
\tikzset{every show grid/.style={show grid/keep bb}}%  Keep the original bounding box!
%\tikzset{grid/.style={gray,very thin,opacity=1}}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usetikzlibrary{
	arrows,%
	arrows.meta,%
	backgrounds,%
	calc,%
	intersections,%
	decorations.markings,%
	decorations.pathmorphing,%
	decorations.pathreplacing,%
	fadings,arrows,%
	fit,%
	petri,%
	positioning,%
	shadows,%
}

\usepgfplotslibrary{
	colormaps,%
	patchplots,%
}


\ifpgffigurefinal
	\tikzset{helpgrid/.style={show grid={left,above,false}}}
\else
	\tikzset{helpgrid/.style={show grid={left,above,true}}}
\fi

\pgfplotsset{every axis/.append style={%
		scaled x ticks=false,%
		scaled y ticks=false,
		/pgf/number format/fixed, 
%		/pgf/number format/1000 sep={\septhouthand},
	},
}

\ifpgffigurecyr
	\tikzset{/pgf/number format/use comma,%
		/pgf/number format/1000 sep={},
	}
\fi

\endinput